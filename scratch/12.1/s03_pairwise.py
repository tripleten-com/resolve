import json
from resolve.llm import ask, MODEL_REVIEW

def strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        text = text.rsplit("```", 1)[0]
    return text.strip()

PAIRWISE = """Two assistants drafted a reply to the same Quillstack support ticket. Which draft is better?
Better means correct and supported by the retrieved chunks. Length is not quality.

Ticket:
{ticket}

Retrieved chunks:
{chunks}

Draft A:
{a}

Draft B:
{b}

Reply with JSON only, keys in this order: {{"reasoning": "<two sentences>", "winner": "A" or "B"}}"""

def winner(ticket: str, chunks: str, a: str, b: str) -> str:
    raw = ask(PAIRWISE.format(ticket=ticket, chunks=chunks, a=a, b=b), model=MODEL_REVIEW, max_tokens=300)
    return json.loads(strip_fences(raw))["winner"]

def compare(ticket: str, chunks: str, current: str, candidate: str) -> str:
    global first, second
    first = winner(ticket, chunks, current, candidate)     # candidate in slot B
    second = winner(ticket, chunks, candidate, current)    # candidate in slot A
    if first == "B" and second == "A":
        return "candidate"
    if first == "A" and second == "B":
        return "current"
    return "tie"

states = json.load(open("evals/results/lesson_drafts.json", encoding="utf-8"))
s = states["T-10038"]
ticket = f"Subject: {s['ticket']['subject']}\n{s['ticket']['body']}"
chunks = "\n\n".join(f"[{d['chunk_id']}] {d['title']}\n{d['text']}" for d in s["docs"])
short = ("Hi Zara,\n\nThat charge is the prorated cost of the seat you added mid-cycle: on the Team plan a seat is "
         "$12 a month and you pay only for the days left in the cycle. Settings > Billing > Invoices shows the "
         "exact calculation.\n\nQuillstack Support")
print(compare(ticket, chunks, current=s["draft"], candidate=short)); print("first:", first, "second:", second)
