"""Check that the configured model endpoint supports what Resolve needs.

    python scripts/check_provider.py            # uses LLM_* from .env
    python scripts/check_provider.py --model Qwen/Qwen3-235B-A22B-Instruct-2507

Five checks, each a real call with a tiny prompt:
  1. plain chat completion and usage counts
  2. tool calling with tool_choice="auto" (does the model ask for the tool?)
  3. forced tool choice (the classifier and the reviewer rely on this)
  4. response_format json_schema (structured output)
  5. the review model, if different, on check 1

Prints PASS / FAIL per check with the raw reason, so a new provider or model can be judged in a minute.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

TOOL = {
    "type": "function",
    "function": {
        "name": "search_kb",
        "description": "Search the Quillstack help center. Use it for any product question.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}, "k": {"type": "integer", "minimum": 1, "maximum": 10}},
            "required": ["query"],
        },
    },
}
SCHEMA = {
    "name": "classification",
    "schema": {
        "type": "object",
        "properties": {
            "type": {"type": "string", "enum": ["how_to", "billing", "bug", "security"]},
            "urgency": {"type": "string", "enum": ["low", "normal", "high"]},
            "aimed_at_agent": {"type": "boolean"},
            "confidence": {"type": "number"},
        },
        "required": ["type", "urgency", "aimed_at_agent", "confidence"],
        "additionalProperties": False,
    },
}
TICKET = "Subject: QS-1042 on every request since this morning\nBody: Our CI pipeline started failing with QS-1042. Nothing changed on our side. Team plan."


def run(name: str, fn):
    t0 = time.perf_counter()
    try:
        detail = fn()
        print(f"PASS  {name}  ({(time.perf_counter() - t0) * 1000:.0f} ms)  {detail}")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL  {name}  {type(exc).__name__}: {str(exc)[:300]}")
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("LLM_MODEL"))
    ap.add_argument("--review-model", default=os.environ.get("LLM_MODEL_REVIEW"))
    args = ap.parse_args()
    base = os.environ.get("LLM_BASE_URL")
    key = os.environ.get("LLM_API_KEY")
    if not (base and key and args.model):
        print("Set LLM_BASE_URL, LLM_API_KEY and LLM_MODEL in .env (or pass --model).")
        return 2
    client = OpenAI(base_url=base, api_key=key)
    print(f"endpoint {base}\nmodel    {args.model}\nreview   {args.review_model or '(same)'}\n")
    ok = []

    def chat():
        r = client.chat.completions.create(model=args.model, max_tokens=20, temperature=0,
                                           messages=[{"role": "user", "content": "Reply with the single word: ready"}])
        u = r.usage
        return f"reply={r.choices[0].message.content!r} tokens={getattr(u, 'prompt_tokens', '?')}/{getattr(u, 'completion_tokens', '?')}"
    ok.append(run("chat completion", chat))

    def tools_auto():
        r = client.chat.completions.create(
            model=args.model, max_tokens=200, temperature=0, tools=[TOOL], tool_choice="auto",
            messages=[{"role": "system", "content": "You are Quillstack's support agent. Use the tool before answering product questions."},
                      {"role": "user", "content": TICKET}])
        calls = r.choices[0].message.tool_calls or []
        if not calls:
            raise RuntimeError(f"no tool call; model answered directly: {r.choices[0].message.content!r}")
        args_ = json.loads(calls[0].function.arguments)
        return f"called {calls[0].function.name} with {args_}"
    ok.append(run("tool calling (auto)", tools_auto))

    def tools_forced():
        r = client.chat.completions.create(
            model=args.model, max_tokens=200, temperature=0, tools=[TOOL],
            tool_choice={"type": "function", "function": {"name": "search_kb"}},
            messages=[{"role": "user", "content": "Say hello."}])
        calls = r.choices[0].message.tool_calls or []
        if not calls or calls[0].function.name != "search_kb":
            raise RuntimeError("forced tool choice ignored")
        json.loads(calls[0].function.arguments)
        return "forced call honoured with valid JSON arguments"
    ok.append(run("tool calling (forced)", tools_forced))

    def json_schema():
        r = client.chat.completions.create(
            model=args.model, max_tokens=200, temperature=0,
            response_format={"type": "json_schema", "json_schema": SCHEMA},
            messages=[{"role": "system", "content": "Classify the ticket. Return JSON matching the schema: "
                                                    + json.dumps(SCHEMA["schema"])},
                      {"role": "user", "content": TICKET}])
        text = r.choices[0].message.content or ""
        text = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        data = json.loads(text)
        for k in SCHEMA["schema"]["required"]:
            if k not in data:
                raise RuntimeError(f"missing key {k}: {data}")
        return f"parsed {data}"
    ok.append(run("response_format json_schema", json_schema))

    if args.review_model and args.review_model != args.model:
        def review_chat():
            r = client.chat.completions.create(model=args.review_model, max_tokens=20, temperature=0,
                                               messages=[{"role": "user", "content": "Reply with the single word: ready"}])
            return f"reply={r.choices[0].message.content!r}"
        ok.append(run("review model chat", review_chat))

    print(f"\n{sum(ok)}/{len(ok)} checks passed")
    return 0 if all(ok) else 1


if __name__ == "__main__":
    sys.exit(main())
