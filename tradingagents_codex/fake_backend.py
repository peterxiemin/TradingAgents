"""Deterministic test/demo inference. It never calls Codex or a network service."""

from __future__ import annotations

import asyncio
import json


class FakeBackend:
    async def generate(self, *, role: str, prompt: str, output_schema: dict) -> str:
        await asyncio.sleep(0)  # Exercise the same scheduling path as live inference.
        payload = json.loads(prompt.split("RESEARCH_DATA_BEGIN\n", 1)[1].rsplit(
            "\nRESEARCH_DATA_END", 1)[0])
        evidence = [item["id"] for item in payload["evidence"]]
        decision = role in ("research_manager", "portfolio_manager")
        proposal = role in ("trader", "portfolio_manager")
        return json.dumps({
            "role": role,
            "summary": f"DEMO ONLY: deterministic {role} output over synthetic evidence; "
                       "no model call, real investment analysis or order was performed.",
            "evidence_ids": evidence,
            "risks": ["Synthetic demonstration; not evidence about actual markets"],
            "confidence": "low", "stance": "neutral",
            "recommendation": "Hold" if decision else None,
            "action": "Hold" if proposal else None,
        })
