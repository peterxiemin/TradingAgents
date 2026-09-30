# TradingAgents · Codex-only fork

A financial **research** workflow powered by the official Python Codex SDK and a small, explicit Python orchestrator. The default runtime does **not** use LangChain or LangGraph.

Forked from [TauricResearch/TradingAgents](https://github.com/TauricResearch/TradingAgents) at `8b22d43d01d9ddda5d686d093d5385884622f3de` (v0.5.2), retaining Apache-2.0 licensing. The [original README](README.upstream.md) and legacy implementation remain available for comparison.

## Quick start: fully offline

Python 3.11+:

```bash
python -m venv .venv
source .venv/bin/activate
pip install '.[dev]'
tradingagents --demo --output-dir codex-results
```

Or use `python -m tradingagents_codex --demo`. No arguments display help rather than making a model call.

`--demo` uses deterministic synthetic evidence and fake inference. It exercises the **same orchestrator, output validation and reporting path** as live runs. Its Hold result is a fixture, **not a real investment recommendation**. It makes no network or Codex calls and needs no login or API key.

Each run writes `snapshot.json`, `report.json` and `report.md` in a fresh run directory. Demo reports are prominently labeled. No branch of this program connects to a broker or executes an order.

## Live Codex research

The dependency is the official [`openai-codex` Python SDK](https://learn.chatgpt.com/docs/codex-sdk), pinned to tested version `0.159.2` and its matching runtime. The security boundary intentionally rejects unreviewed versions. Published SDK builds supply their own matching Codex executable; this is not an OpenAI API SDK disguised as a Codex adapter.

First configure a **dedicated Codex home** through the supported Codex authentication flow yourself. This application does not discover, copy, print or create credentials, and does not reuse the assistant's internal session. Account/model access and usage limits still apply. Never paste credentials into chat or commit them to this repository.

```bash
tradingagents --live --ticker SPY --date 2026-09-30 \
  --codex-home /absolute/path/to/your/dedicated-codex-home \
  --output-dir codex-results
```

Optionally pass `--model YOUR_AVAILABLE_CODEX_MODEL`. Leaving it unset uses the configured account default. Live use sends the public research snapshot to the signed-in Codex service and consumes that account's applicable quota/billing. Choose and authorize the account before running it.

Useful limits:

```bash
tradingagents --live --ticker AAPL --date 2026-09-30 \
  --codex-home /absolute/path/to/your/dedicated-codex-home \
  --debate-rounds 1 --risk-rounds 1 --concurrency 4 --attempts 2 \
  --role-timeout 120 --run-timeout 1200 --data-timeout 60 --language Chinese
```

These example dates are examples, not a scheduled job. Supply the actual as-of date you intend to research. This implementation does not provide a web UI or persistent hosting.

## Architecture

1. The host collects one bounded, dated data snapshot and computes its SHA-256 fingerprint
2. Four analysts run concurrently: market, fundamentals, news and sentiment. Each sees only its owned evidence category
3. Bull and bear alternate for a bounded number of rounds
4. A research manager weighs the debate; a trader produces a hypothetical proposal
5. Aggressive, conservative and neutral risk reviewers challenge it in bounded rounds
6. A portfolio manager emits the final five-tier research rating and a consistent hypothetical Buy/Hold/Sell action

The default is 12 inference calls before retries. Debate/risk rounds are capped at three, concurrency at four and attempts at three. There are per-role and overall timeouts, a character budget for prompts, output size/schema validation, evidence-ID ownership checks, cancellation propagation and a stage audit trail. Failures do not silently turn into successful decisions.

- `tradingagents_codex/contracts.py`: typed configuration, role outputs and reports
- `snapshot.py`: immutable evidence, as-of safeguards and host-owned data collection
- `orchestrator.py`: explicit asyncio workflow; the application owns all transitions
- `prompts.py`: adapted upstream analyst, debate, trader and risk objectives
- `codex_backend.py`: official SDK session/process boundary
- `fake_backend.py`: deterministic offline inference
- `cli.py`: CLI, mode selection and saved reports

### Data and tool ownership

Models do not choose tools or fetch data independently. The host fixes the ticker/date and collects an evidence snapshot before inference. Existing Yahoo/SEC data helpers and technical indicators are reused where appropriate; missing or historically unavailable data is labeled. Social sentiment is not invented when no dated social source is supplied.

Every role call starts an isolated ephemeral Codex session in a private working directory. The adapter uses a sanitized subprocess environment, explicit read-only sandbox, deny-all approvals and supported capability controls. Unexpected tool activity fails the call. The SDK does not expose a universal guarantee that the model's declared tool list is empty: read-only sandboxing and denied capabilities are the execution boundary, not merely a prompt promise.

The default CLI never auto-loads `.env` files. A live account directory is accepted only through explicit `--codex-home`. Outputs may still contain public-source text and model errors in reasoning; review research conclusions before using them. Schema validation does not prove financial accuracy.

## Tests

```bash
pip install '.[dev]'
pytest -q tests_codex
ruff check .
```

The new suite covers the complete offline flow, concurrent analyst stage, role ordering, shared snapshot, invalid citations/JSON, retries, timeouts, cancellation, SDK request contracts, CLI output and independence from LangChain/LangGraph. Live inference is intentionally not part of offline CI. There are no live credentials in fixtures.

Legacy comparison:

```bash
pip install '.[legacy,dev]'
pytest -q tests
tradingagents-legacy --help
# Original source CLI remains: python -m cli.main
```

The `legacy` extra alone enables the original LangChain/LangGraph providers. Original backtesting, portfolio-file sizing, memory/reflection and checkpoint resume remain on that legacy path; they are **not claimed as migrated**. New runs currently collect a fresh snapshot and restart after failure, with no checkpoint resume. No historical performance or trading-profit claims are made.

## Docker

```bash
docker compose build
docker compose run --rm tradingagents --demo
```

The default container is the Codex-only runtime. Compose's named results volume retains its reports. Live authentication needs an explicitly configured dedicated account directory, kept **outside the Docker build context** and mounted only at runtime. Known Codex homes, auth files, dotenv secrets, results and virtual environments are excluded from the build context. Docker instructions are provided for portability, not a claim that Docker was exercised in the current cloud environment.

## Verification boundary

Offline tests can establish orchestration and packaging correctness. Actual Codex login/model access, live inference, third-party data availability and live end-to-end quality must be verified separately with an authorized account. GitHub Actions are intentionally disabled on this personal fork; local checks do not imply a green remote CI run.
