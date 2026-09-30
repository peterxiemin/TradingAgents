"""A finite asyncio state machine. No graph framework and no model-owned tools."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Callable
from datetime import UTC, datetime

from pydantic import ValidationError

from .contracts import AuditEvent, Backend, ResearchError, Role, RoleOutput, RunConfig, RunReport
from .prompts import ANALYSTS, RISK_ROLES, build_prompt
from .snapshot import Snapshot

_ACTION_FOR_RATING = {
    "Buy": "Buy", "Overweight": "Buy", "Hold": "Hold", "Underweight": "Sell", "Sell": "Sell",
}


class Orchestrator:
    def __init__(self, backend: Backend, config: RunConfig | None = None,
                 on_event: Callable[[AuditEvent], None] | None = None):
        self.backend = backend
        self.config = config or RunConfig()
        self.on_event = on_event
        self._running = False

    async def run(self, snapshot: Snapshot) -> RunReport:
        if self._running:
            raise ResearchError("An orchestrator instance cannot run two analyses concurrently")
        # Validate before reserving the instance, so malformed caller state cannot
        # leave the runner permanently marked busy.
        validated = Snapshot.model_validate_json(snapshot.model_dump_json())
        if validated.mode == "live" and not any(
            item.category == "market" and item.status == "available" for item in validated.evidence
        ):
            raise ResearchError("Live research requires usable market evidence; no model calls started")
        self._running = True
        self._audit: list[AuditEvent] = []
        self._semaphore = asyncio.Semaphore(self.config.max_concurrency)
        self._calls = 0
        # Deep validation creates our own frozen copy rather than sharing caller state.
        self._snapshot = validated
        original_hash = self._snapshot.fingerprint()
        try:
            async with asyncio.timeout(self.config.run_timeout_seconds):
                self._emit("analysts_started")
                async with asyncio.TaskGroup() as group:
                    tasks = [group.create_task(self._role(role, (), 1)) for role in ANALYSTS]
                outputs = [task.result() for task in tasks]
                self._emit("analysts_completed")
                for round_number in range(1, self.config.debate_rounds + 1):
                    for role in ("bull", "bear"):
                        outputs.append(await self._role(role, tuple(outputs), round_number))
                outputs.append(await self._role("research_manager", tuple(outputs), 1))
                outputs.append(await self._role("trader", tuple(outputs), 1))
                for round_number in range(1, self.config.risk_rounds + 1):
                    for role in RISK_ROLES:
                        outputs.append(await self._role(role, tuple(outputs), round_number))
                final = await self._role("portfolio_manager", tuple(outputs), 1)
                outputs.append(final)
                if self._snapshot.fingerprint() != original_hash:
                    raise ResearchError("Shared snapshot changed during analysis")
                self._emit("run_completed")
                return RunReport(
                    run_id=uuid.uuid4().hex, mode=self._snapshot.mode,
                    ticker=self._snapshot.ticker, as_of=self._snapshot.as_of,
                    generated_at=datetime.now(UTC), snapshot_hash=original_hash,
                    config=self.config, outputs=tuple(outputs), final_decision=final,
                    audit=tuple(self._audit),
                )
        except TimeoutError as exc:
            self._emit("run_failed", detail="Overall timeout")
            raise ResearchError("Overall run timeout; all active roles were cancelled") from exc
        except asyncio.CancelledError:
            self._emit("run_cancelled")
            raise
        except Exception as exc:
            self._emit("run_failed", detail=type(exc).__name__)
            if isinstance(exc, ResearchError):
                raise
            raise ResearchError("Research failed; no completed decision was produced") from exc
        finally:
            self._running = False

    def _emit(self, event: str, **kwargs) -> None:
        item = AuditEvent(sequence=len(self._audit) + 1, event=event, **kwargs)
        self._audit.append(item)
        if self.on_event is not None:
            self.on_event(item)

    async def _role(self, role: Role, prior: tuple[RoleOutput, ...],
                    round_number: int) -> RoleOutput:
        feedback = ""
        async with self._semaphore:
            for attempt in range(1, self.config.max_attempts + 1):
                self._calls += 1
                if self._calls > self.config.max_role_calls * self.config.max_attempts:
                    raise ResearchError("Role call budget exceeded")
                self._emit("role_started", role=role, round=round_number, attempt=attempt)
                try:
                    prompt = build_prompt(
                        role, self._snapshot, prior, language=self.config.language,
                        round_number=round_number, validation_feedback=feedback,
                    )
                    if len(prompt) > self.config.max_prompt_chars:
                        raise ResearchError("Prompt budget exceeded; reduce rounds or evidence size")
                    async with asyncio.timeout(self.config.role_timeout_seconds):
                        raw = await self.backend.generate(
                            role=role, prompt=prompt, output_schema=RoleOutput.model_json_schema(),
                        )
                    if not isinstance(raw, str) or len(raw) > 64_000:
                        raise ValueError("Role response must be JSON text smaller than 64KB")
                    result = RoleOutput.model_validate_json(raw)
                    self._validate_output(role, result)
                    self._emit("role_completed", role=role, round=round_number, attempt=attempt)
                    return result
                except (ValidationError, ValueError, TimeoutError, RuntimeError) as exc:
                    self._emit("role_retry" if attempt < self.config.max_attempts else "role_failed",
                               role=role, round=round_number, attempt=attempt,
                               detail=type(exc).__name__)
                    # No raw provider errors or model output in retry prompts/logs: they
                    # can contain secrets, transport headers or prompt injections.
                    feedback = (
                        "The prior attempt failed output validation or timed out. Return valid "
                        "JSON, the exact requested role, only supplied evidence IDs, and all "
                        "required recommendation/action fields."
                    )
                    if attempt == self.config.max_attempts:
                        raise ResearchError(f"Role {role} failed after {attempt} attempts") from exc
        raise AssertionError("unreachable")

    def _validate_output(self, role: Role, result: RoleOutput) -> None:
        if result.role != role:
            raise ValueError("Role identity mismatch")
        permitted = {
            item.id for item in self._snapshot.evidence
            if role not in ANALYSTS or item.category == role
        }
        if not set(result.evidence_ids) <= permitted:
            raise ValueError("Unknown or unowned evidence citation")
        if role in ("research_manager", "portfolio_manager") and result.recommendation is None:
            raise ValueError("A decision role must provide a five-tier recommendation")
        if role in ("trader", "portfolio_manager") and result.action is None:
            raise ValueError("A proposal/final role must provide an action")
        if role == "portfolio_manager" and result.action != _ACTION_FOR_RATING[result.recommendation]:
            raise ValueError("Final action conflicts with the recommendation")
        if self._snapshot.mode == "live" and role in (
            "research_manager", "trader", "portfolio_manager",
        ):
            usable = {item.id for item in self._snapshot.evidence if item.status == "available"}
            if not set(result.evidence_ids) & usable:
                raise ValueError("Decision roles must cite at least one usable source")
