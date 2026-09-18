"""Shared OpenRouter client for the dataset generators.

Reads OPENROUTER_API_KEY from resolve/.env (never printed, never written out).
Every call appends one line to data/_build/usage_log.jsonl with token counts
and an estimated cost so total spend can be audited.
"""
from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
USAGE_LOG = DATA / "_build" / "usage_log.jsonl"
MODEL = "anthropic/claude-sonnet-4.5"
# Published OpenRouter prices for the generation model, USD per token.
PRICE_IN = 3.0 / 1_000_000
PRICE_OUT = 15.0 / 1_000_000

load_dotenv(ROOT / ".env")
_client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])
_lock = threading.Lock()


def chat(messages: list[dict], *, tag: str, max_tokens: int = 8000, temperature: float = 0.8, retries: int = 4) -> str:
    """One chat completion. Returns the text; logs usage."""
    last_err = None
    for attempt in range(retries):
        try:
            t0 = time.time()
            resp = _client.chat.completions.create(
                model=MODEL,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                extra_body={"usage": {"include": True}},
            )
            text = resp.choices[0].message.content or ""
            u = resp.usage
            rec = {
                "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "tag": tag,
                "model": MODEL,
                "prompt_tokens": getattr(u, "prompt_tokens", 0),
                "completion_tokens": getattr(u, "completion_tokens", 0),
                "cost_usd": round(getattr(u, "prompt_tokens", 0) * PRICE_IN + getattr(u, "completion_tokens", 0) * PRICE_OUT, 5),
                "reported_cost_usd": getattr(u, "cost", None),
                "latency_s": round(time.time() - t0, 1),
                "finish_reason": resp.choices[0].finish_reason,
            }
            with _lock:
                USAGE_LOG.parent.mkdir(parents=True, exist_ok=True)
                with USAGE_LOG.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(rec) + "\n")
            return text
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"LLM call failed after {retries} attempts: {last_err}")


def usage_summary() -> dict:
    tot_in = tot_out = 0
    cost = 0.0
    n = 0
    if USAGE_LOG.exists():
        for line in USAGE_LOG.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            n += 1
            tot_in += r["prompt_tokens"]
            tot_out += r["completion_tokens"]
            cost += r["cost_usd"]
    return {"calls": n, "prompt_tokens": tot_in, "completion_tokens": tot_out, "cost_usd": round(cost, 3)}


if __name__ == "__main__":
    print(json.dumps(usage_summary(), indent=2))
