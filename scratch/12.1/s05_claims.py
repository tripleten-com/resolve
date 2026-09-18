import json
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import BaseModel, Field
from resolve.llm import chat, MODEL_REVIEW

class Claim(BaseModel):
    claim: str
    supported: bool
    evidence: str | None = Field(default=None, description="chunk_id of the supporting chunk, or null")

class Claims(BaseModel):
    """Every factual claim in the draft and whether a retrieved chunk supports it."""
    claims: list[Claim]

SYSTEM = ("Split the draft into its individual factual claims (a number, a limit, a step, a menu path, an error code, a "
          "promise about what happens next). Greetings and sign-offs are not claims. For each claim say whether one of "
          "the retrieved chunks supports it, naming the chunk_id. A claim that only restates the customer's message "
          "is supported. Be literal: 'up to 24 hours' is not supported by a chunk that says '48 hours'.")

def format_chunks(docs: list[dict]) -> str:
    return "\n\n".join(f"[{d['chunk_id']}] {d['title']}\n{d['text']}" for d in docs) or "(nothing retrieved)"

def groundedness(state: dict) -> dict:
    schema = convert_to_openai_tool(Claims)
    resp = chat([{"role": "system", "content": SYSTEM},
                 {"role": "user", "content": f"Retrieved chunks:\n{format_chunks(state['docs'])}\n\nDRAFT:\n{state['draft']}\n\nCall Claims."}],
                model=MODEL_REVIEW, tools=[schema], max_tokens=900,
                tool_choice={"type": "function", "function": {"name": "Claims"}})
    c = Claims.model_validate_json(resp.choices[0].message.tool_calls[0].function.arguments)
    n, supported = len(c.claims), sum(x.supported for x in c.claims)
    return {"claims": [x.model_dump() for x in c.claims], "n_claims": n, "supported": supported,
            "share_supported": round(supported / n, 4) if n else None,
            "hallucination": n > 0 and supported < n}

states = json.load(open("evals/results/lesson_drafts.json", encoding="utf-8"))
results = {tid: groundedness(s) for tid, s in states.items()}
for tid, g in results.items():
    print(tid, f"{g['supported']}/{g['n_claims']} supported, share {g['share_supported']}, hallucination {g['hallucination']}")
    for c in g["claims"]:
        if not c["supported"]:
            print("    unsupported:", c["claim"])
shares = [g["share_supported"] for g in results.values()]
print("mean share supported:", round(sum(shares) / len(shares), 3))
print("hallucination rate:", sum(g["hallucination"] for g in results.values()), "/", len(results))
json.dump(results, open("scratch/12.1/claims.json", "w"), indent=2)
