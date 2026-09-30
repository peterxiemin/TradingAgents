"""All orchestration checks are deterministic, isolated and network-free."""

import asyncio
import json

import pytest
from pydantic import ValidationError

from tradingagents_codex.contracts import ResearchError, RoleOutput, RunConfig
from tradingagents_codex.fake_backend import FakeBackend
from tradingagents_codex.orchestrator import Orchestrator
from tradingagents_codex.snapshot import demo_snapshot


class RecordingBackend(FakeBackend):
    def __init__(self, delay=0):
        self.calls = []
        self.prompts = []
        self.active = 0
        self.peak = 0
        self.cancelled = 0
        self.delay = delay

    async def generate(self, **kwargs):
        self.calls.append(kwargs["role"])
        self.prompts.append(kwargs["prompt"])
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            await asyncio.sleep(self.delay)
            return await super().generate(**kwargs)
        except asyncio.CancelledError:
            self.cancelled += 1
            raise
        finally:
            self.active -= 1


def test_whole_flow_and_shared_evidence():
    backend = RecordingBackend(0.001)
    snapshot = demo_snapshot()
    report = asyncio.run(Orchestrator(backend).run(snapshot))
    assert len(report.outputs) == 12
    assert report.final_decision.role == "portfolio_manager"
    assert report.final_decision.action == "Hold"
    assert report.execution == "research_only_no_orders"
    assert report.mode == "demo"
    assert backend.peak == 4
    assert backend.calls[4:] == ["bull", "bear", "research_manager", "trader",
                                "risk_aggressive", "risk_conservative", "risk_neutral",
                                "portfolio_manager"]
    assert all(snapshot.fingerprint() in prompt for prompt in backend.prompts)
    assert report.snapshot_hash == snapshot.fingerprint()


def test_rounds_and_concurrency_are_bounded():
    backend = RecordingBackend(0.001)
    config = RunConfig(debate_rounds=3, risk_rounds=2, max_concurrency=2)
    report = asyncio.run(Orchestrator(backend, config).run(demo_snapshot()))
    assert len(report.outputs) == config.max_role_calls == 19
    assert backend.peak == 2
    assert backend.calls.count("bull") == 3
    assert backend.calls.count("risk_conservative") == 2


@pytest.mark.parametrize("fields", [
    {"debate_rounds": 0}, {"risk_rounds": 4}, {"max_concurrency": 5},
    {"max_attempts": 4}, {"role_timeout_seconds": 0}, {"run_timeout_seconds": 8000},
    {"max_prompt_chars": 1}, {"language": ""},
])
def test_config_rejects_unbounded_values(fields):
    with pytest.raises(ValidationError):
        RunConfig(**fields)


def test_bad_json_retried_with_fresh_request():
    class Recovering(RecordingBackend):
        async def generate(self, **kwargs):
            raw = await super().generate(**kwargs)
            if kwargs["role"] == "bull" and self.calls.count("bull") == 1:
                return "not json"
            return raw

    backend = Recovering()
    report = asyncio.run(Orchestrator(backend).run(demo_snapshot()))
    assert backend.calls.count("bull") == 2
    assert len(report.outputs) == 12
    assert any(event.event == "role_retry" for event in report.audit)


@pytest.mark.parametrize("invalid", ["wrong_role", "invented_source", "wrong_action", "no_action"])
def test_invalid_decisions_fail_closed(invalid):
    class Invalid(FakeBackend):
        async def generate(self, **kwargs):
            data = json.loads(await super().generate(**kwargs))
            if kwargs["role"] == "portfolio_manager":
                if invalid == "wrong_role":
                    data["role"] = "bull"
                elif invalid == "invented_source":
                    data["evidence_ids"] = ["nonexistent"]
                elif invalid == "wrong_action":
                    data["action"] = "Buy"
                else:
                    data["action"] = None
            return json.dumps(data)

    with pytest.raises(ResearchError):
        asyncio.run(Orchestrator(Invalid(), RunConfig(max_attempts=1)).run(demo_snapshot()))


def test_analysts_can_only_cite_their_owned_category():
    class WrongOwner(FakeBackend):
        async def generate(self, **kwargs):
            data = json.loads(await super().generate(**kwargs))
            if kwargs["role"] == "market":
                data["evidence_ids"] = [next(item.id for item in demo_snapshot().evidence
                                             if item.category == "news")]
            return json.dumps(data)

    with pytest.raises(ResearchError):
        asyncio.run(Orchestrator(WrongOwner(), RunConfig(max_attempts=1)).run(demo_snapshot()))


def test_role_timeout_cancels_work_and_stops_pipeline():
    backend = RecordingBackend(0.05)
    config = RunConfig(role_timeout_seconds=0.001, max_attempts=1)
    with pytest.raises(ResearchError):
        asyncio.run(Orchestrator(backend, config).run(demo_snapshot()))
    assert backend.active == 0
    assert backend.cancelled > 0
    assert "trader" not in backend.calls


def test_overall_timeout_cancels_all_tasks():
    backend = RecordingBackend(0.05)
    with pytest.raises(ResearchError, match="Overall run timeout"):
        asyncio.run(Orchestrator(backend, RunConfig(run_timeout_seconds=0.005)).run(demo_snapshot()))
    assert backend.active == 0
    assert backend.cancelled == 4


def test_caller_cancellation_propagates():
    async def check():
        backend = RecordingBackend(1)
        task = asyncio.create_task(Orchestrator(backend).run(demo_snapshot()))
        await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert backend.active == 0
        assert backend.cancelled == 4
    asyncio.run(check())


def test_independent_runs_have_no_prior_role_state():
    async def check():
        backend = RecordingBackend()
        runner = Orchestrator(backend)
        first = await runner.run(demo_snapshot())
        second = await runner.run(demo_snapshot())
        assert first.run_id != second.run_id
        assert len(first.outputs) == len(second.outputs) == 12
        assert first.audit[0].sequence == second.audit[0].sequence == 1
        assert '"prior_reports":[]' in backend.prompts[12]
    asyncio.run(check())


def test_output_rejects_unknown_fields_and_invalid_enum():
    raw = {
        "role": "bull", "summary": "test", "evidence_ids": [], "risks": [],
        "confidence": "low", "stance": "neutral", "recommendation": None, "action": None,
    }
    with pytest.raises(ValidationError):
        RoleOutput.model_validate(raw | {"execute_order": True})
    with pytest.raises(ValidationError):
        RoleOutput.model_validate(raw | {"action": "BUY NOW"})


def test_invalid_snapshot_does_not_poison_runner():
    async def check():
        runner = Orchestrator(FakeBackend())
        invalid = demo_snapshot().model_copy(update={"evidence": ()})
        with pytest.raises(ValidationError):
            await runner.run(invalid)
        assert (await runner.run(demo_snapshot())).final_decision.role == "portfolio_manager"
    asyncio.run(check())


def test_live_missing_market_data_stops_before_inference():
    backend = RecordingBackend()
    snapshot = demo_snapshot()
    snapshot = snapshot.model_copy(update={
        "mode": "live",
        "evidence": tuple(item.model_copy(update={"status": "unavailable"})
                          for item in snapshot.evidence),
    })
    with pytest.raises(ResearchError, match="requires usable market evidence"):
        asyncio.run(Orchestrator(backend).run(snapshot))
    assert backend.calls == []


@pytest.mark.parametrize("citations", [[], ["company_profile"]])
def test_live_decision_must_cite_available_evidence(citations):
    class Ungrounded(FakeBackend):
        async def generate(self, **kwargs):
            data = json.loads(await super().generate(**kwargs))
            if kwargs["role"] == "research_manager":
                data["evidence_ids"] = citations
            return json.dumps(data)
    snapshot = demo_snapshot().model_copy(update={"mode": "live"})
    with pytest.raises(ResearchError):
        asyncio.run(Orchestrator(Ungrounded(), RunConfig(max_attempts=1)).run(snapshot))


def test_usage_totals_preserve_sdk_semantics_and_run_isolation():
    class Metered(FakeBackend):
        def __init__(self):
            self.usage_records = []

        async def generate(self, **kwargs):
            raw = await super().generate(**kwargs)
            self.usage_records.append({
                "role": kwargs["role"], "model": "test-model", "input_tokens": 100,
                "cached_input_tokens": 40, "output_tokens": 30,
                "reasoning_output_tokens": 10, "total_tokens": 130,
            })
            return raw

    async def check():
        runner = Orchestrator(Metered())
        for _ in range(2):
            result = await runner.run(demo_snapshot())
            assert len(result.role_token_usage) == 12
            assert result.token_totals.total_tokens == 1560
            assert result.token_totals.output_tokens == 360
            assert result.token_totals.reasoning_output_tokens == 120
            assert result.token_usage_complete is True
    asyncio.run(check())


def test_missing_usage_is_unknown_not_zero():
    result = asyncio.run(Orchestrator(FakeBackend()).run(demo_snapshot()))
    assert result.token_totals is None
    assert result.token_usage_complete is False


def test_shared_backend_rejects_overlapping_runs_and_releases_afterwards():
    async def check():
        backend = RecordingBackend(0.01)
        one, two = Orchestrator(backend), Orchestrator(backend)
        first = asyncio.create_task(one.run(demo_snapshot()))
        await asyncio.sleep(0.002)
        with pytest.raises(ResearchError, match="only one active research run"):
            await two.run(demo_snapshot())
        await first
        assert len((await two.run(demo_snapshot())).outputs) == 12
    asyncio.run(check())


def test_backend_ownership_released_after_cancellation():
    async def check():
        backend = RecordingBackend(0.01)
        task = asyncio.create_task(Orchestrator(backend).run(demo_snapshot()))
        await asyncio.sleep(0.002)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert len((await Orchestrator(backend).run(demo_snapshot())).outputs) == 12
    asyncio.run(check())
