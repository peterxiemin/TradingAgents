"""Offline snapshot contract, isolation and point-in-time regression tests."""

from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
import sys
import time
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from tradingagents_codex import snapshot as snap


@pytest.fixture
def legacy(monkeypatch):
    """No dotenv discovery or network is permitted by any offline test."""
    import dotenv
    import requests

    monkeypatch.setattr(dotenv, "load_dotenv", lambda *args, **kwargs: False)
    monkeypatch.setattr(dotenv, "find_dotenv", lambda *args, **kwargs: "")

    def no_network(*args, **kwargs):
        raise AssertionError("Network is forbidden in this offline test")

    monkeypatch.setattr(requests, "get", no_network)
    monkeypatch.setattr(requests, "head", no_network)
    monkeypatch.setattr(requests.Session, "request", no_network)
    return monkeypatch


def test_demo_is_deterministic_frozen_and_clearly_synthetic():
    one, two = snap.demo_snapshot(), snap.demo_snapshot()
    assert one == two
    assert one.fingerprint() == two.fingerprint()
    assert len(one.fingerprint()) == 64
    canonical = json.dumps(
        one.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    assert one.fingerprint() == hashlib.sha256(canonical.encode()).hexdigest()
    assert one.mode == "demo"
    assert one.as_of == snap.DEMO_DATE
    assert "SYNTHETIC DEMO ONLY" in one.evidence[0].content
    assert all(e.status == "unavailable" for e in one.evidence[1:])
    with pytest.raises(ValidationError):
        one.ticker = "AAPL"
    with pytest.raises(ValidationError):
        one.evidence[0].content = "changed"


def test_demo_hash_changes_with_ticker_or_date_and_has_no_future_rows():
    example = snap.demo_snapshot("aapl", date(2025, 2, 1))
    assert example.ticker == "AAPL"
    assert example.fingerprint() != snap.demo_snapshot("MSFT", example.as_of).fingerprint()
    assert example.fingerprint() != snap.demo_snapshot("AAPL", date(2025, 2, 2)).fingerprint()
    market = json.loads(example.evidence[0].content)
    assert all(date.fromisoformat(row["date"]) <= example.as_of for row in market["recent_ohlcv"])
    assert market["latest_ohlcv"]["date"] == "2025-01-31"


@pytest.mark.parametrize(
    "ticker",
    [
        "",
        "  ",
        "..",
        "../AAPL",
        "/tmp/a",
        "AAPL;id",
        "AAPL\nMSFT",
        "A" * 33,
        "$(id)",
        "💸",
        "A..B",
        None,
    ],
)
def test_reject_invalid_tickers(ticker):
    with pytest.raises(ValueError):
        snap.demo_snapshot(ticker)
    with pytest.raises(ValueError):
        snap.collect_snapshot(ticker, date(2025, 1, 1))


@pytest.mark.parametrize(
    "ticker", ["SPY", "BRK-B", "^GSPC", "BTC-USD", "EURUSD=X", "0700.HK", "GC=F"]
)
def test_accept_common_vendor_tickers(ticker):
    assert snap.demo_snapshot(ticker).ticker == ticker


@pytest.mark.parametrize(
    "day",
    [
        date(1899, 12, 31),
        datetime.now(UTC).date() + timedelta(days=1),
        "2025-01-01",
        datetime(2025, 1, 1, tzinfo=UTC),
    ],
)
def test_reject_invalid_analysis_dates(day):
    with pytest.raises(ValueError):
        snap.demo_snapshot(as_of=day)
    with pytest.raises(ValueError):
        snap.collect_snapshot("SPY", day)


@pytest.mark.parametrize("timeout", [0, -1, 601, float("nan"), float("inf"), True, "1"])
def test_reject_invalid_timeout(timeout):
    with pytest.raises(ValueError):
        snap.collect_snapshot("SPY", date(2025, 1, 1), timeout_seconds=timeout)


def test_model_validation_bounds_and_duplicate_ids():
    example = snap.demo_snapshot()
    raw = example.model_dump()
    raw["captured_at"] = datetime(2025, 1, 1)
    with pytest.raises(ValidationError):
        snap.Snapshot.model_validate(raw)
    raw = example.model_dump()
    raw["evidence"] = (example.evidence[0], example.evidence[0])
    with pytest.raises(ValidationError):
        snap.Snapshot.model_validate(raw)
    with pytest.raises(ValidationError):
        snap.Evidence(
            id="bad id", category="market", source="test", content="x", status="available"
        )
    with pytest.raises(ValidationError):
        snap.Evidence(
            id="one",
            category="market",
            source="test",
            content="x" * (snap.MAX_CONTENT_CHARS + 1),
            status="available",
        )
    evidence = tuple(
        snap.Evidence(
            id=str(i),
            category="market",
            source="test",
            content="x" * snap.MAX_CONTENT_CHARS,
            status="available",
        )
        for i in range(8)
    )
    with pytest.raises(ValidationError):
        snap.Snapshot(
            ticker="SPY",
            as_of=date(2025, 1, 1),
            captured_at=datetime.now(UTC),
            mode="live",
            evidence=evidence,
        )
    clipped = snap._bounded("🚀" * 10_000)
    assert len(clipped) == snap.MAX_CONTENT_CHARS
    assert "TRUNCATED" in clipped


def test_worker_failure_is_explicit_not_invented(monkeypatch):
    monkeypatch.setattr(
        snap.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout="", returncode=1)
    )
    result = snap.collect_snapshot("SPY", date(2025, 1, 1))
    assert result.mode == "live"
    assert len(result.evidence) == len(snap._SPECS)
    assert {e.status for e in result.evidence} == {"unavailable", "withheld"}
    assert next(e for e in result.evidence if e.id == "company_profile").status == "withheld"


def test_timeout_retains_complete_partial_results(monkeypatch):
    good = snap._item(snap._SPECS[0], "Dated market result", "available")

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(
            args[0], 0.05, output=(good.model_dump_json() + "\n{partial").encode()
        )

    monkeypatch.setattr(snap.subprocess, "run", timeout)
    result = snap.collect_snapshot("SPY", date(2025, 1, 1), timeout_seconds=0.05)
    assert result.evidence[0] == good
    assert "deadline exceeded" in result.evidence[2].content
    assert result.evidence[2].status == "unavailable"


def test_os_start_failure_is_unavailable(monkeypatch):
    def failed(*args, **kwargs):
        raise OSError("sensitive path must never be exposed")

    monkeypatch.setattr(snap.subprocess, "run", failed)
    result = snap.collect_snapshot("SPY", date(2025, 1, 1))
    assert "could not start" in result.evidence[0].content
    assert "sensitive" not in result.model_dump_json()


def test_worker_env_and_cache_paths_are_isolated(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "do-not-forward")
    monkeypatch.setenv("HOME", "/private-home-never-use")
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "0")
    calls = []

    def fake_run(args, **kwargs):
        request = json.loads(kwargs["input"])
        calls.append(request["run_dir"])
        assert Path(request["run_dir"]).is_dir()
        assert kwargs["cwd"] == request["run_dir"]
        assert kwargs["env"]["HOME"] == request["run_dir"]
        assert kwargs["env"]["PYTHON_DOTENV_DISABLED"] == "1"
        assert "OPENAI_API_KEY" not in kwargs["env"]
        assert kwargs["timeout"] == 2
        return SimpleNamespace(stdout="", returncode=0)

    monkeypatch.setattr(snap.subprocess, "run", fake_run)
    snap.collect_snapshot("SPY", date(2025, 1, 1), timeout_seconds=2)
    snap.collect_snapshot("SPY", date(2025, 1, 1), timeout_seconds=2)
    assert len(set(calls)) == 2
    assert all(not Path(path).exists() for path in calls)
    assert os.environ["HOME"] == "/private-home-never-use"
    assert os.environ["PYTHON_DOTENV_DISABLED"] == "0"


def test_worker_transport_requires_explicit_validated_mapping(monkeypatch, tmp_path):
    monkeypatch.setenv("HTTPS_PROXY", "http://secret:password@localhost:1234")
    assert "HTTPS_PROXY" not in snap._worker_env(str(tmp_path))
    values = {"HTTPS_PROXY": "http://127.0.0.1:4321"}
    assert snap._worker_env(str(tmp_path), values)["HTTPS_PROXY"] == values["HTTPS_PROXY"]
    with pytest.raises(ValueError):
        snap._worker_env(str(tmp_path), {"HTTPS_PROXY": "http://secret:password@localhost:1234"})


def test_collect_passes_only_validated_transport_to_subprocess(monkeypatch):
    def fake_run(args, **kwargs):
        assert kwargs["env"]["HTTPS_PROXY"] == "http://127.0.0.1:4321"
        assert "OPENAI_API_KEY" not in kwargs["env"]
        return SimpleNamespace(stdout="", returncode=0)
    monkeypatch.setattr(snap.subprocess, "run", fake_run)
    snap.collect_snapshot("SPY", date(2025, 1, 1), transport_env={
        "HTTPS_PROXY": "http://127.0.0.1:4321",
    })


def test_protocol_rejects_wrong_identity_malformed_and_oversized_output():
    good = snap._item(snap._SPECS[0], "result", "available")
    bad = good.model_copy(update={"source": "Unapproved source"})
    assert snap._decode_worker(bad.model_dump_json()) == {}
    assert snap._decode_worker("x" * (snap.MAX_WORKER_BYTES + 1)) == {}
    assert snap._decode_worker("not json\n" + good.model_dump_json()) == {good.id: good}


def test_snapshot_import_does_not_import_legacy_or_llm_packages(tmp_path):
    script = """
import sys
from tradingagents_codex.snapshot import demo_snapshot
assert demo_snapshot().mode == 'demo'
assert not any(m == 'tradingagents' or m.startswith(('langchain', 'langgraph', 'dotenv')) for m in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=snap._worker_env(str(tmp_path)),
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr


def test_legacy_imports_are_keyless_and_never_import_llm_packages(tmp_path):
    # Imports are exercised in a real fresh subprocess, but readers never run.
    script = """
import builtins, io, sys
from datetime import date
original_open = builtins.open
original_io_open = io.open
def checked_open(file, *args, **kwargs):
    assert '.env' not in str(file) and 'auth.json' not in str(file), str(file)
    return original_open(file, *args, **kwargs)
def checked_io_open(file, *args, **kwargs):
    assert '.env' not in str(file) and 'auth.json' not in str(file), str(file)
    return original_io_open(file, *args, **kwargs)
builtins.open = checked_open
io.open = checked_io_open
from tradingagents_codex.snapshot import _legacy_tools
readers = _legacy_tools('SPY', date(2025, 1, 1))
assert len(readers) == 5
assert not any(m.startswith(('langchain', 'langgraph')) for m in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=snap._worker_env(str(tmp_path)),
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr


def test_legacy_imports_refuse_to_run_in_host(monkeypatch):
    monkeypatch.delenv("TRADINGAGENTS_CODEX_DATA_WORKER", raising=False)
    with pytest.raises(RuntimeError, match="isolated worker"):
        snap._legacy_tools("SPY", date(2025, 1, 1))


def test_market_adapter_removes_future_rows_before_indicators(legacy):
    import pandas as pd
    import yfinance as yf

    dates = pd.bdate_range("2024-09-01", "2025-01-15")
    frame = pd.DataFrame(
        {"Open": 100.0, "High": 101.0, "Low": 99.0, "Close": 100.0, "Volume": 1000.0}, index=dates
    )
    frame.index.name = "Date"
    frame.loc[pd.Timestamp("2025-01-15"), ["Open", "High", "Low", "Close"]] = 999999.0
    history_calls = []

    class FakeTicker:
        def __init__(self, symbol):
            assert symbol == "SPY"

        def history(self, **kwargs):
            history_calls.append(kwargs)
            return frame

    legacy.setattr(yf, "Ticker", FakeTicker)
    result = json.loads(snap._market_content("SPY", date(2025, 1, 14)))
    assert result["latest_ohlcv"]["close"] == 100
    assert result["indicators"]["close_50_sma"] == 100
    assert result["indicators"]["close_200_sma"] is None
    assert all(row["date"] <= "2025-01-14" for row in result["recent_ohlcv"])
    assert history_calls[0]["end"] == "2025-01-15"
    assert history_calls[0]["auto_adjust"] is False
    assert len(result["recent_ohlcv"]) == 20


def test_news_adapter_filters_future_and_undated_items(legacy):
    import yfinance as yf

    articles = [
        {
            "title": "valid",
            "providerPublishTime": datetime(2025, 1, 14, 10, tzinfo=UTC).timestamp(),
            "summary": "x" * 2000,
        },
        {"title": "future", "providerPublishTime": datetime(2025, 1, 15, tzinfo=UTC).timestamp()},
        {"title": "undated"},
        {"content": {"title": "also valid", "pubDate": "2025-01-14T01:00:00Z"}},
        {"title": "old", "providerPublishTime": datetime(2024, 1, 14, tzinfo=UTC).timestamp()},
    ]
    legacy.setattr(yf, "Ticker", lambda ticker: SimpleNamespace(get_news=lambda **kwargs: articles))
    result = json.loads(snap._news_content("SPY", date(2025, 1, 14)))
    assert [a["title"] for a in result["articles"]] == ["valid", "also valid"]
    assert len(result["articles"][0]["summary"]) == 500
    legacy.setattr(yf, "Ticker", lambda ticker: SimpleNamespace(get_news=lambda **kwargs: []))
    with pytest.raises(ValueError, match="No dated news"):
        snap._news_content("SPY", date(2025, 1, 14))


def test_sec_upstream_reuses_filed_date_vintages(legacy):
    from tradingagents.dataflows.vendors import sec_edgar

    facts = {
        "Assets": {
            "units": {
                "USD": [
                    {"end": "2024-09-30", "filed": "2024-11-01", "val": 100, "form": "10-Q"},
                    {"end": "2024-09-30", "filed": "2025-02-01", "val": 999, "form": "10-Q/A"},
                ]
            }
        }
    }
    values, unit = sec_edgar._as_of(facts, ("Assets",), "2025-01-15", ((60, 115),))
    assert values == {("2024-09-30", 0): 100}
    assert unit == "USD"


def test_worker_tool_failure_and_size_limits(legacy, tmp_path):
    import yfinance as yf

    def failed():
        raise RuntimeError("secret=do-not-log")

    legacy.setattr(
        snap,
        "_legacy_tools",
        lambda *args: (
            (snap._SPECS[0], failed),
            (snap._SPECS[2], lambda: "x" * 100_000),
            (snap._SPECS[5], lambda: ""),
        ),
    )
    legacy.setattr(yf, "set_tz_cache_location", lambda path: None)
    legacy.setattr(
        sys,
        "stdin",
        io.StringIO(json.dumps({"ticker": "SPY", "as_of": "2025-01-15", "run_dir": str(tmp_path)})),
    )
    output = io.StringIO()
    legacy.setattr(sys, "stdout", output)
    snap._worker_main()
    parsed = snap._decode_worker(output.getvalue())
    assert parsed["market_snapshot"].status == "unavailable"
    assert "RuntimeError" in parsed["market_snapshot"].content
    assert "do-not-log" not in output.getvalue()
    assert len(parsed["balance_sheet"].content) == snap.MAX_CONTENT_CHARS
    assert parsed["company_news"].status == "unavailable"


def test_actual_hard_timeout_kills_worker(monkeypatch):
    actual_run = subprocess.run

    def sleeping_worker(args, **kwargs):
        return actual_run([sys.executable, "-c", "import time; time.sleep(30)"], **kwargs)

    monkeypatch.setattr(snap.subprocess, "run", sleeping_worker)
    started = time.monotonic()
    result = snap.collect_snapshot("SPY", date(2025, 1, 1), timeout_seconds=0.05)
    assert time.monotonic() - started < 3
    assert result.evidence[0].status == "unavailable"
    assert "deadline exceeded" in result.evidence[0].content
