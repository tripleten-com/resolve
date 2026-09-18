"""Token, cost, and latency accounting: every node adds what it spent to state['usage']."""
from __future__ import annotations

import time

from resolve.config import graph_config
from resolve.llm import usage_of


def cost_usd(resp, model: str) -> float:
    """Cost of one response: the provider's own figure if it sends one, else the pricing table."""
    u = getattr(resp, "usage", None)
    provider_cost = getattr(u, "cost", None) if u is not None else None
    if provider_cost is None and u is not None and hasattr(u, "model_extra"):
        provider_cost = (u.model_extra or {}).get("cost")
    if isinstance(provider_cost, (int, float)):
        return float(provider_cost)
    prices = graph_config().get("pricing", {})
    p = prices.get(model) or prices.get("default", {"input": 1.0, "output": 5.0})
    tokens = usage_of(resp)
    return tokens["input_tokens"] / 1e6 * p["input"] + tokens["output_tokens"] / 1e6 * p["output"]


class Meter:
    """Collects the spend of one node: `meter.add(resp, model)` per call, `meter.merge(state)` at the end."""

    def __init__(self) -> None:
        self.t0 = time.perf_counter()
        self.input_tokens = 0
        self.output_tokens = 0
        self.cost = 0.0

    def add(self, resp, model: str) -> None:
        u = usage_of(resp)
        self.input_tokens += u["input_tokens"]
        self.output_tokens += u["output_tokens"]
        self.cost += cost_usd(resp, model)

    def merge(self, state: dict) -> dict:
        prev = state.get("usage") or {}
        return {
            "input_tokens": int(prev.get("input_tokens", 0)) + self.input_tokens,
            "output_tokens": int(prev.get("output_tokens", 0)) + self.output_tokens,
            "cost_usd": round(float(prev.get("cost_usd", 0.0)) + self.cost, 6),
            "latency_ms": int(prev.get("latency_ms", 0)) + int((time.perf_counter() - self.t0) * 1000),
        }
