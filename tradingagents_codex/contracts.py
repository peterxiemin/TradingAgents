"""Validated boundaries between independent research roles."""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

Rating = Literal["Buy", "Overweight", "Hold", "Underweight", "Sell"]
Role = Literal[
    "market", "fundamentals", "news", "sentiment", "bull", "bear",
    "research_manager", "trader", "risk_aggressive", "risk_conservative",
    "risk_neutral", "portfolio_manager",
]


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RunConfig(FrozenModel):
    debate_rounds: int = Field(default=1, ge=1, le=3)
    risk_rounds: int = Field(default=1, ge=1, le=3)
    max_concurrency: int = Field(default=4, ge=1, le=4)
    max_attempts: int = Field(default=2, ge=1, le=3)
    role_timeout_seconds: float = Field(default=120, gt=0, le=600)
    run_timeout_seconds: float = Field(default=1200, gt=0, le=7200)
    max_prompt_chars: int = Field(default=250_000, ge=10_000, le=1_000_000)
    language: str = Field(default="Chinese", min_length=1, max_length=40)

    @property
    def max_role_calls(self) -> int:
        return 7 + 2 * self.debate_rounds + 3 * self.risk_rounds


class RoleOutput(FrozenModel):
    role: Role
    summary: str = Field(min_length=1, max_length=8000)
    evidence_ids: tuple[str, ...] = Field(max_length=64)
    risks: tuple[str, ...] = Field(max_length=20)
    confidence: Literal["low", "medium", "high"]
    stance: Literal["bullish", "bearish", "neutral"]
    recommendation: Rating | None
    action: Literal["Buy", "Hold", "Sell"] | None


class AuditEvent(FrozenModel):
    sequence: int
    event: str
    role: str | None = None
    round: int | None = None
    attempt: int | None = None
    detail: str | None = None


class TokenCounters(FrozenModel):
    """SDK counters; cached/reasoning are subsets, not extra billable additions."""

    input_tokens: int | None = Field(default=None, ge=0)
    cached_input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    reasoning_output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)


class RoleTokenUsage(TokenCounters):
    role: Role
    model: str | None = None


class RunReport(FrozenModel):
    format_version: Literal[1] = 1
    run_id: str
    mode: Literal["demo", "live"]
    ticker: str
    as_of: date
    generated_at: datetime
    snapshot_hash: str
    config: RunConfig
    outputs: tuple[RoleOutput, ...]
    final_decision: RoleOutput
    audit: tuple[AuditEvent, ...]
    role_token_usage: tuple[RoleTokenUsage, ...] = ()
    token_totals: TokenCounters | None = None
    token_usage_complete: bool = False
    execution: Literal["research_only_no_orders"] = "research_only_no_orders"


class Backend(Protocol):
    async def generate(self, *, role: str, prompt: str, output_schema: dict) -> str: ...


class ResearchError(RuntimeError):
    """A bounded run failed; a partial result is never a completed decision."""
