"""The one model client every part of Resolve uses.

Reads three values from the environment (a .env file is loaded if present):

    LLM_BASE_URL      the OpenAI-compatible endpoint, e.g. https://api.tokenfactory.nebius.com/v1/
    LLM_API_KEY       your key for that endpoint
    LLM_MODEL         the model for classifying and for the specialists
    LLM_MODEL_REVIEW  the model for the reviewer and the judge (optional; defaults to LLM_MODEL)

Any provider that speaks the OpenAI chat API works. The course route is Nebius Token Factory;
OpenRouter, OpenAI, Anthropic's compatibility endpoint, or a local server such as Ollama work too.

Run `python -m resolve.llm --check` to confirm the key and model work.
"""
from __future__ import annotations

import os
import sys
import time

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.tokenfactory.nebius.com/v1/")
API_KEY = os.environ.get("LLM_API_KEY", "")
MODEL = os.environ.get("LLM_MODEL", "Qwen/Qwen3-235B-A22B-Instruct-2507")
MODEL_REVIEW = os.environ.get("LLM_MODEL_REVIEW", MODEL)

_client: OpenAI | None = None


def client() -> OpenAI:
    """A single shared client, created on first use."""
    global _client
    if _client is None:
        if not API_KEY:
            raise RuntimeError("LLM_API_KEY is not set. Copy .env.example to .env and fill it in.")
        _client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
    return _client


def chat(messages: list[dict], *, model: str | None = None, temperature: float = 0.0,
         max_tokens: int = 800, tools: list[dict] | None = None, **kwargs):
    """One chat completion. Returns the raw response so callers can read tool calls and usage."""
    params = dict(model=model or MODEL, messages=messages, temperature=temperature,
                  max_tokens=max_tokens, **kwargs)
    if tools:
        params["tools"] = tools
    return client().chat.completions.create(**params)


def ask(prompt: str, *, system: str | None = None, model: str | None = None,
        temperature: float = 0.0, max_tokens: int = 800) -> str:
    """The simplest call: one prompt in, the reply text out."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    resp = chat(messages, model=model, temperature=temperature, max_tokens=max_tokens)
    return resp.choices[0].message.content or ""


def usage_of(resp) -> dict:
    """Token counts from a response, as a plain dict (zeros if the provider sent none)."""
    u = getattr(resp, "usage", None)
    return {
        "input_tokens": getattr(u, "prompt_tokens", 0) or 0,
        "output_tokens": getattr(u, "completion_tokens", 0) or 0,
    }


def _check() -> int:
    if not API_KEY:
        print("LLM_API_KEY is not set. Copy .env.example to .env and fill it in.")
        return 1
    t0 = time.perf_counter()
    try:
        resp = chat([{"role": "user", "content": "Reply with the single word: ready"}], max_tokens=10)
    except Exception as exc:  # noqa: BLE001
        print(f"Call failed against {BASE_URL} with model {MODEL}: {exc}")
        return 1
    ms = (time.perf_counter() - t0) * 1000
    text = (resp.choices[0].message.content or "").strip()
    u = usage_of(resp)
    print(f"{MODEL} via {BASE_URL}: {text!r} in {ms:.0f} ms "
          f"({u['input_tokens']} in / {u['output_tokens']} out tokens)")
    if MODEL_REVIEW != MODEL:
        resp2 = chat([{"role": "user", "content": "Reply with the single word: ready"}],
                     model=MODEL_REVIEW, max_tokens=10)
        print(f"{MODEL_REVIEW} (review): {(resp2.choices[0].message.content or '').strip()!r}")
    return 0


if __name__ == "__main__":
    if "--check" in sys.argv:
        sys.exit(_check())
    print(__doc__)
