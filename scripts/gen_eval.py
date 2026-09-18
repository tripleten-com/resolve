"""Generate the retrieval evaluation sets with the LLM.

Usage: python scripts/gen_eval.py

Writes data/eval/retrieval_dev.jsonl (~80) and data/eval/retrieval_heldout.jsonl (~40):
{"query": "...", "gold_doc_ids": ["kb-0045"]}. Queries are phrased as customers
phrase things; some hinge on an exact error code or config key. Raw model output
is cached in data/_build/retrieval_raw.json.
"""
from __future__ import annotations

import json
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from llm import DATA, chat, usage_summary  # noqa: E402

RAW = DATA / "_build" / "retrieval_raw.json"
N_DEV, N_HELDOUT = 80, 40
SEED = 7

# Articles that overlap enough that a query can legitimately have both as gold.
PAIRS = [
    ("kb-0044", "kb-0045"), ("kb-0045", "kb-0046"), ("kb-0066", "kb-0067"), ("kb-0068", "kb-0069"),
    ("kb-0031", "kb-0032"), ("kb-0034", "kb-0035"), ("kb-0024", "kb-0025"), ("kb-0059", "kb-0060"),
    ("kb-0064", "kb-0065"), ("kb-0049", "kb-0050"), ("kb-0050", "kb-0051"), ("kb-0006", "kb-0007"),
    ("kb-0003", "kb-0016"), ("kb-0029", "kb-0030"), ("kb-0020", "kb-0021"), ("kb-0036", "kb-0037"),
    ("kb-0052", "kb-0046"), ("kb-0052", "kb-0055"), ("kb-0004", "kb-0033"), ("kb-0054", "kb-0064"),
]

SYSTEM = """You write search queries that real Quillstack customers would type into a help-center search box or a support form. Quillstack is a B2B SaaS product for team documentation and project tracking (workspaces, pages, projects/tasks, API keys, SSO on Business, Slack/GitHub integrations, import/export, mobile app; plans Free, Team, Business).

For each article you are given, write exactly 2 queries that the article answers:
- Query A: plain customer phrasing, 4 to 14 words, may be a question or a fragment, never a copy of the title, no title-case.
- Query B: a different angle: either (1) it contains the exact error code (QS-1xxx), config key (like sso.enforce, api.ip_allowlist, slack.default_channel), header name, or menu path that appears in the article's facts, when the article has one; or (2) it describes the symptom in everyday words that the documentation does not use (for example "token stopped working" instead of "API key expired", "charged for days I didn't use" instead of "proration"), sometimes with a typo.
- gold_doc_ids: the article id, plus the paired article id ONLY when the query is answered equally well by both (pairs are listed). Never more than 2 ids.

OUTPUT: a JSON array of objects {"query": str, "gold_doc_ids": [str, ...]} in article order, 2 per article. No commentary, no code fences."""


def main() -> None:
    facts = {f["id"]: f["text"] for f in yaml.safe_load((DATA / "facts.yaml").read_text(encoding="utf-8"))["facts"]}
    index = json.loads((DATA / "kb" / "index.json").read_text(encoding="utf-8"))
    public = [it for it in index if it["space"] == "public"]
    public_ids = {it["id"] for it in public}
    pair_text = "; ".join(f"{a}+{b}" for a, b in PAIRS)

    if RAW.exists():
        raw = json.loads(RAW.read_text(encoding="utf-8"))
    else:
        batches = [public[i:i + 15] for i in range(0, len(public), 15)]

        def run(batch):
            lines = [f"PAIRS: {pair_text}", "", "ARTICLES:"]
            for it in batch:
                lines.append(f"\n{it['id']}: {it['title']} [labels: {', '.join(it['labels'])}]")
                for fid in it["facts"]:
                    lines.append(f"  - {facts[fid]}")
            text = chat([{"role": "system", "content": SYSTEM}, {"role": "user", "content": "\n".join(lines)}],
                        tag=f"eval:{batch[0]['id']}", max_tokens=6000, temperature=0.9)
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.S)
            arr = json.loads(text[text.find("["):text.rfind("]") + 1])
            # attach the intended article to each pair of queries by position
            out = []
            for i, it in enumerate(batch):
                for q in arr[2 * i:2 * i + 2]:
                    out.append({"query": q["query"], "gold_doc_ids": q.get("gold_doc_ids") or [it["id"]], "source": it["id"]})
            return out

        raw = []
        with ThreadPoolExecutor(max_workers=3) as ex:
            for got in ex.map(run, batches):
                raw.extend(got)
        RAW.write_text(json.dumps(raw, indent=1, ensure_ascii=False), encoding="utf-8")

    # hand corrections after review: query text -> gold ids
    FIXES = {
        "share anyone with the link option missing": ["kb-0010"],
        "webhook stopped working and says it's disabled": ["kb-0051", "kb-0075"],
        "accidentally deleted a page how do I get it back": ["kb-0007", "kb-0070"],
    }
    # clean: gold ids must exist, must include the source article, 1..2 ids, dedupe queries
    seen = set()
    rows = []
    for r in raw:
        if r["query"] in FIXES:
            r["gold_doc_ids"] = FIXES[r["query"]]
            r["source"] = FIXES[r["query"]][0]
        gold = [g for g in r["gold_doc_ids"] if g in public_ids]
        if r["source"] not in gold:
            gold.insert(0, r["source"])
        gold = gold[:2]
        q = re.sub(r"\s+", " ", r["query"]).strip()
        key = q.lower()
        if not q or key in seen:
            continue
        seen.add(key)
        rows.append({"query": q, "gold_doc_ids": gold, "source": r["source"]})

    # split: keep both queries of an article in the same split where possible
    rng = random.Random(SEED)
    by_src: dict[str, list[dict]] = {}
    for r in rows:
        by_src.setdefault(r["source"], []).append(r)
    srcs = list(by_src)
    rng.shuffle(srcs)
    dev, held = [], []
    for s in srcs:
        target = held if len(held) < N_HELDOUT else dev
        target.extend(by_src[s])
    rng.shuffle(dev)
    rng.shuffle(held)
    dev = dev[:N_DEV]
    held = held[:N_HELDOUT]
    (DATA / "eval").mkdir(exist_ok=True)
    for name, rs in (("retrieval_dev", dev), ("retrieval_heldout", held)):
        with (DATA / "eval" / f"{name}.jsonl").open("w", encoding="utf-8", newline="\n") as f:
            for r in rs:
                f.write(json.dumps({"query": r["query"], "gold_doc_ids": r["gold_doc_ids"]}, ensure_ascii=False) + "\n")
        codes = sum(bool(re.search(r"QS-1\d{3}|\b\w+\.\w+_\w+\b|\bsso\.\w+|api\.\w+|slack\.\w+", r["query"])) for r in rs)
        print(f"{name}: {len(rs)} queries, {codes} with an exact code/key, {sum(len(r['gold_doc_ids']) == 2 for r in rs)} with 2 gold ids")
    print("usage:", usage_summary())


if __name__ == "__main__":
    main()
