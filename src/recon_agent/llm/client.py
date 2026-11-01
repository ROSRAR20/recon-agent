"""LLM client interface, mock client implementation, and cost tracker.

Provides:
- LLMClient: Protocol for LLM invocation.
- MockClient: Deterministic replay client for offline testing.
- CostTracker: Token estimator and cost calculation for Gemini models.
"""

from __future__ import annotations

import json
from typing import Any, Protocol

from src.recon_agent.llm.bundle import CaseBundle
from src.recon_agent.llm.cache import LLMCache


class LLMClient(Protocol):
    """Protocol for LLM client implementations."""

    def complete(self, bundle: CaseBundle) -> str:
        """Produce completion string for a CaseBundle."""
        ...


class CostTracker:
    """Estimates token counts and cost for supported models."""

    # Rates per 1 million tokens (USD)
    RATES = {
        "gemini-2.5-pro": {"input": 1.25, "output": 5.00},
        "gemini-2.5-flash": {"input": 0.075, "output": 0.30},
    }

    @classmethod
    def estimate_tokens(cls, text: str) -> int:
        """Heuristic token estimation (~4 characters per token)."""
        return max(1, len(text) // 4)

    @classmethod
    def calculate_cost(cls, model_id: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate estimated USD cost."""
        rates = cls.RATES.get(model_id, {"input": 0.0, "output": 0.0})
        cost = (input_tokens * rates["input"] + output_tokens * rates["output"]) / 1_000_000.0
        return round(cost, 6)


class MockClient:
    """Deterministic mock client for offline tests and replaying canned responses."""

    def __init__(
        self,
        canned_responses: dict[str, str] | None = None,
        model_id: str = "mock-recon-v1",
        cache: LLMCache | None = None,
    ) -> None:
        self.canned_responses = canned_responses or {}
        self.model_id = model_id
        self.cache = cache
        self.call_count = 0

    def default_safe_response(self, bundle: CaseBundle) -> str:
        """Default safe escalate_human response when no canned response matches."""
        resp = {
            "decision": "escalate_human",
            "predicted_class": None,
            "reason_code": "AMBIGUOUS_TEXT",
            "explanation": f"Ambiguous discrepancy for entity {bundle.entity_id}; escalating for human review.",
            "evidence": [{"field": "entity_id", "value": bundle.entity_id}],
            "cited_amount_paise": None,
            "verbatim_quote": None,
        }
        return json.dumps(resp)

    def complete(self, bundle: CaseBundle, prompt_hash: str = "mock_prompt") -> str:
        """Complete bundle using cache, canned response, or safe default."""
        self.call_count += 1
        bundle_hash = bundle.bundle_hash

        # Check cache if available
        if self.cache is not None:
            cached = self.cache.get(bundle_hash, prompt_hash, self.model_id)
            if cached is not None:
                return cached

        # Lookup canned response or fallback to safe response
        if bundle_hash in self.canned_responses:
            raw = self.canned_responses[bundle_hash]
        elif bundle.entity_id in self.canned_responses:
            raw = self.canned_responses[bundle.entity_id]
        else:
            raw = self.default_safe_response(bundle)

        if self.cache is not None:
            getattr(self.cache, "put")(bundle_hash, prompt_hash, self.model_id, raw)

        return raw
