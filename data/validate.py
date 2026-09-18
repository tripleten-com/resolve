"""Re-check every referential rule in the Resolve dataset and print counts.

Usage: python data/validate.py        (exit code 1 if any hard error)
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

import yaml

DATA = Path(__file__).resolve().parent
errors: list[str] = []
warnings: list[str] = []


def err(msg: str) -> None:
    errors.append(msg)


def warn(msg: str) -> None:
    warnings.append(msg)


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


# ---------------- facts ----------------
facts_doc = yaml.safe_load((DATA / "facts.yaml").read_text(encoding="utf-8"))
facts = {f["id"]: f["text"] for f in facts_doc["facts"]}
if not 40 <= len(facts) <= 60:
    err(f"facts.yaml has {len(facts)} facts; expected 40..60")
for i, fid in enumerate(facts, 1):
    if fid != f"F{i:03d}":
        err(f"fact id {fid} out of sequence")


def anchors(text: str) -> list[str]:
    """Distinctive strings a fact contributes: error codes, config keys, numbers with units, paths."""
    a = set(re.findall(r"QS-1\d{3}", text))
    a |= set(re.findall(r"\b[a-z]+\.[a-z_]+\b", text))               # sso.enforce, api.ip_allowlist
    a |= set(re.findall(r"\b\d[\d,]*\s?(?:days?|minutes?|hours?|seconds?|GB|MB|business days?|business hours|CIDR)", text))
    a |= set(re.findall(r"\$\d+", text))
    a |= set(re.findall(r"\b(?:INV-|qs_live_|whsec_|X-Quillstack-Signature|Retry-After|next_cursor|page_size|request_id|HMAC-SHA256)\S*", text))
    a |= set(re.findall(r"security@quillstack\.com|app\.quillstack\.com|api\.quillstack\.com/v2", text))
    a |= set(re.findall(r"\b(?:at most|up to)\s+\d[\d,]*", text))
    a |= set(re.findall(r"\b\d+\b", text))
    return sorted(a, key=len, reverse=True)


# ---------------- knowledge base ----------------
index = json.loads((DATA / "kb" / "index.json").read_text(encoding="utf-8"))
by_id = {it["id"]: it for it in index}
public_ids = {it["id"] for it in index if it["space"] == "public"}
internal_ids = {it["id"] for it in index if it["space"] == "internal"}
FM_RE = re.compile(r"^---\nid: (kb-\d{4})\ntitle: (.+)\nspace: (public|internal)\nlabels: \[(.*)\]\nupdated: (\d{4}-\d{2}-\d{2})\nversion: (\d+)\n---\n", re.M)
word_counts: dict[str, int] = {}
fact_uses: Counter = Counter()
files_seen = set()
for space in ("public", "internal"):
    for p in sorted((DATA / "kb" / space).glob("*.md")):
        text = p.read_text(encoding="utf-8")
        m = FM_RE.match(text)
        if not m:
            err(f"{p.name}: front matter malformed")
            continue
        doc_id, title, sp, labels, updated, version = m.groups()
        files_seen.add(doc_id)
        if doc_id != p.stem:
            err(f"{p.name}: id {doc_id} does not match filename")
        if sp != space:
            err(f"{p.name}: space {sp} but in folder {space}")
        it = by_id.get(doc_id)
        if not it:
            err(f"{p.name}: not in index.json")
            continue
        if it["title"] != title or it["space"] != sp:
            err(f"{p.name}: title/space differ from index.json")
        body = text[m.end():]
        wc = len(re.findall(r"\S+", body))
        word_counts[doc_id] = wc
        if not 500 <= wc <= 900:
            warn(f"{doc_id}: body has {wc} words (target 500..900)")
        for h in ("## Before you start", "## Steps", "## Troubleshooting", "## Related"):
            if h not in body:
                warn(f"{doc_id}: missing section '{h}'")
        if not (re.search(r"^\|.*\|$", body, re.M) or "```" in body):
            warn(f"{doc_id}: no table or code block")
        for link in re.findall(r"\]\((kb-\d{4})\)", body):
            if link not in by_id:
                err(f"{doc_id}: links to unknown article {link}")
            elif space == "public" and link in internal_ids:
                err(f"{doc_id}: public article links to internal page {link}")
        # each covered fact must leave at least one anchor string in the body
        for fid in it["facts"]:
            if fid not in facts:
                err(f"{doc_id}: covers unknown fact {fid}")
                continue
            fact_uses[fid] += 1
            anc = anchors(facts[fid])
            if anc and not any(a in body for a in anc):
                warn(f"{doc_id}: no anchor of {fid} found in body ({anc[:3]}...)")
for it in index:
    if it["id"] not in files_seen:
        err(f"{it['id']} listed in index.json but no file")
unused = [f for f in facts if fact_uses[f] == 0]
if unused:
    warn(f"facts never covered by an article: {unused}")

# injection seeds must be present verbatim in the named internal articles
seeds = json.loads((DATA / "kb" / "injection_seeds.json").read_text(encoding="utf-8"))
for s in seeds:
    p = DATA / s["path"]
    if not p.exists() or s["text"] not in p.read_text(encoding="utf-8"):
        err(f"injection seed for {s['doc_id']} not found verbatim in {s['path']}")
    if s["doc_id"] not in internal_ids:
        err(f"injection seed {s['doc_id']} is not an internal page")

# ---------------- accounts ----------------
accounts = {a["email"]: a for a in read_jsonl(DATA / "accounts.jsonl")}
acct_rows = read_jsonl(DATA / "accounts.jsonl")
if len(acct_rows) != len(accounts):
    err("accounts.jsonl has duplicate emails")
for a in acct_rows:
    if a["plan"] not in ("Free", "Team", "Business"):
        err(f"account {a['email']}: bad plan {a['plan']}")
    if len(a["invoices"]) > 3 or len(a["charges"]) > 5:
        err(f"account {a['email']}: too many invoices/charges")
    for inv in a["invoices"]:
        if not re.fullmatch(r"INV-\d{6}-\d{5}", inv["id"]) or inv["id"][4:10] != inv["date"][:4] + inv["date"][5:7]:
            err(f"account {a['email']}: invoice id {inv['id']} does not match date {inv['date']}")
    for ch in a["charges"]:
        if not re.fullmatch(r"ch_[0-9a-f]{12}", ch["id"]):
            err(f"account {a['email']}: bad charge id {ch['id']}")


def has_duplicate_charge(a: dict) -> bool:
    seen = Counter((c["date"], c["amount"]) for c in a["charges"] if c.get("status", "succeeded") == "succeeded")
    return any(n >= 2 for n in seen.values())


# ---------------- tickets ----------------
TICKET_KEYS = ["ticket_id", "created_at", "channel", "customer_email", "plan", "subject", "body", "type", "urgency",
               "expected_action", "gold_doc_ids", "key_facts", "is_injection", "injection_kind"]
TARGET = {"how_to": 0.45, "billing": 0.25, "bug": 0.20, "security": 0.10}
all_ids: Counter = Counter()
split_stats = {}
for split in ("dev", "heldout"):
    rows = read_jsonl(DATA / "tickets" / f"{split}.jsonl")
    types = Counter()
    n_inj = n_unans = n_howto = 0
    emails = set()
    for r in rows:
        if list(r.keys()) != TICKET_KEYS:
            err(f"{r.get('ticket_id')}: keys {list(r.keys())}")
        all_ids[r["ticket_id"]] += 1
        if not re.fullmatch(r"T-1\d{4}", r["ticket_id"]):
            err(f"{r['ticket_id']}: bad ticket id")
        if not re.fullmatch(r"2026-(08|09)-\d{2}T\d{2}:\d{2}:\d{2}Z", r["created_at"]):
            err(f"{r['ticket_id']}: created_at {r['created_at']} not in Aug-Sep 2026")
        if r["channel"] not in ("web", "email"):
            err(f"{r['ticket_id']}: channel {r['channel']}")
        if r["plan"] not in ("Free", "Team", "Business"):
            err(f"{r['ticket_id']}: plan {r['plan']}")
        if r["type"] not in TARGET or r["urgency"] not in ("low", "normal", "high") or r["expected_action"] not in ("auto_resolve", "escalate", "create_issue"):
            err(f"{r['ticket_id']}: bad label enum")
        if not r["subject"].strip() or len(r["body"].split()) < 5:
            err(f"{r['ticket_id']}: empty subject or body")
        for g in r["gold_doc_ids"]:
            if g not in public_ids:
                err(f"{r['ticket_id']}: gold_doc_id {g} is not a public article")
        for k in r["key_facts"]:
            if k not in facts:
                err(f"{r['ticket_id']}: key_fact {k} does not exist")
        if r["is_injection"] != (r["injection_kind"] is not None):
            err(f"{r['ticket_id']}: is_injection/injection_kind mismatch")
        if r["injection_kind"] not in (None, "direct", "indirect", "exfiltration", "excessive_agency", "jailbreak"):
            err(f"{r['ticket_id']}: injection_kind {r['injection_kind']}")
        if r["type"] == "security" and (r["gold_doc_ids"] or r["expected_action"] != "escalate"):
            err(f"{r['ticket_id']}: security ticket must escalate with no gold docs")
        if r["type"] == "bug" and r["expected_action"] != "create_issue":
            err(f"{r['ticket_id']}: bug ticket must create_issue")
        if r["type"] == "how_to" and not r["gold_doc_ids"] and r["expected_action"] != "escalate":
            err(f"{r['ticket_id']}: unanswerable how_to must escalate")
        if r["type"] == "how_to" and r["gold_doc_ids"] and r["expected_action"] != "auto_resolve":
            err(f"{r['ticket_id']}: answerable how_to must auto_resolve")
        a = accounts.get(r["customer_email"])
        if not a:
            err(f"{r['ticket_id']}: no account for {r['customer_email']}")
        else:
            if a["plan"] != r["plan"]:
                err(f"{r['ticket_id']}: plan {r['plan']} differs from account plan {a['plan']}")
            if r["type"] == "billing":
                if "F016" in r["key_facts"] and not has_duplicate_charge(a):
                    err(f"{r['ticket_id']}: duplicate-charge ticket but account has no duplicate charge")
                if "F009" in r["key_facts"] and r["expected_action"] != "escalate":
                    err(f"{r['ticket_id']}: refund ticket must escalate")
                if "F014" in r["key_facts"] and not any(c.get("status") == "failed" for c in a["charges"]):
                    err(f"{r['ticket_id']}: failed-payment ticket but account has no failed charge")
                if "F021" in r["key_facts"] and r["expected_action"] != "escalate":
                    err(f"{r['ticket_id']}: other-customer billing ticket must escalate")
        types[r["type"]] += 1
        n_inj += r["is_injection"]
        if r["type"] == "how_to":
            n_howto += 1
            n_unans += not r["gold_doc_ids"]
        emails.add(r["customer_email"])
    n = len(rows)
    for t, share in TARGET.items():
        if abs(types[t] / n - share) > 0.05:
            err(f"{split}: type {t} share {types[t] / n:.2f} is more than 5 points from {share}")
    split_stats[split] = {"tickets": n, "types": dict(types), "injections": n_inj, "unanswerable_how_to": n_unans,
                          "how_to": n_howto, "distinct_emails": len(emails),
                          "urgency": dict(Counter(r["urgency"] for r in rows)),
                          "expected_action": dict(Counter(r["expected_action"] for r in rows)),
                          "injection_kinds": dict(Counter(r["injection_kind"] for r in rows if r["is_injection"])),
                          "with_logs_or_code": sum(bool(re.search(r"QS-1\d{3}|request_id|Traceback|\d{4}-\d{2}-\d{2}T\d{2}:", r["body"])) for r in rows)}
dups = [t for t, c in all_ids.items() if c > 1]
if dups:
    err(f"duplicate ticket ids across splits: {dups}")
ticket_emails = {r["customer_email"] for s in ("dev", "heldout") for r in read_jsonl(DATA / "tickets" / f"{s}.jsonl")}
extra = set(accounts) - ticket_emails
if extra:
    warn(f"{len(extra)} accounts have no ticket")

# ---------------- retrieval eval ----------------
eval_stats = {}
for name in ("retrieval_dev", "retrieval_heldout"):
    rows = read_jsonl(DATA / "eval" / f"{name}.jsonl")
    qs = Counter(r["query"].lower() for r in rows)
    if any(c > 1 for c in qs.values()):
        err(f"{name}: duplicate queries")
    for r in rows:
        if set(r.keys()) != {"query", "gold_doc_ids"}:
            err(f"{name}: bad keys {list(r.keys())}")
        if not 1 <= len(r["gold_doc_ids"]) <= 2:
            err(f"{name}: query '{r['query'][:40]}' has {len(r['gold_doc_ids'])} gold ids")
        for g in r["gold_doc_ids"]:
            if g not in public_ids:
                err(f"{name}: gold id {g} is not a public article")
    eval_stats[name] = {"queries": len(rows),
                        "with_code_or_key": sum(bool(re.search(r"QS-1\d{3}|\b(?:sso|api|slack)\.[a-z_]+|X-Quillstack-Signature|Retry-After|next_cursor|page_size", r["query"])) for r in rows),
                        "two_gold": sum(len(r["gold_doc_ids"]) == 2 for r in rows)}

# ---------------- injection seed ----------------
seed_rows = read_jsonl(DATA / "eval" / "injection_seed.jsonl")
lab = Counter(r["label"] for r in seed_rows)
if lab["attack"] != 20 or lab["benign"] != 20:
    err(f"injection_seed.jsonl: {dict(lab)}; expected 20 attack and 20 benign")
seed_ids = Counter(r["id"] for r in seed_rows)
if any(c > 1 for c in seed_ids.values()):
    err("injection_seed.jsonl: duplicate ids")
for r in seed_rows:
    if set(r) != {"id", "text", "label", "kind", "expected", "notes"}:
        err(f"injection_seed {r.get('id')}: bad keys")
    if (r["label"] == "attack") != (r["expected"] == "blocked"):
        err(f"injection_seed {r['id']}: label/expected mismatch")

# ---------------- report ----------------
public_wc = [word_counts[i] for i in sorted(public_ids) if i in word_counts]
internal_wc = [word_counts[i] for i in sorted(internal_ids) if i in word_counts]
print("== Resolve dataset validation ==")
print(f"facts: {len(facts)}")
print(f"kb public: {len(public_wc)} articles, {sum(public_wc)} words (min {min(public_wc)}, max {max(public_wc)})")
print(f"kb internal: {len(internal_wc)} pages, {sum(internal_wc)} words (min {min(internal_wc)}, max {max(internal_wc)})")
print(f"kb injection seeds: {len(seeds)} in {[s['doc_id'] for s in seeds]}")
print(f"facts covered by at least one article: {len(facts) - len(unused)}/{len(facts)}")
for split, st in split_stats.items():
    print(f"tickets {split}: {json.dumps(st)}")
print(f"ticket ids unique: {not dups}; total tickets {sum(all_ids.values())}")
print(f"accounts: {len(accounts)} (all ticket emails covered: {ticket_emails <= set(accounts)})")
for name, st in eval_stats.items():
    print(f"eval {name}: {st}")
print(f"eval injection_seed: {dict(lab)}, kinds: {dict(Counter(r['kind'] for r in seed_rows))}")
print(f"warnings: {len(warnings)}")
for w in warnings:
    print("  WARN", w)
print(f"errors: {len(errors)}")
for e in errors:
    print("  ERROR", e)
sys.exit(1 if errors else 0)
