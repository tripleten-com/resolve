# Resolve dataset (Quillstack)

Synthetic data for Project 4 "Resolve", the internal support agent for Quillstack, a B2B SaaS
documentation and project-tracking product with Free, Team and Business plans. Everything in
this folder agrees with one set of facts, `facts.yaml`. Nothing here is real: every company,
person, email address, invoice and charge is invented.

## Files

| Path | What it is | Count |
|---|---|---|
| `facts.yaml` | Canonical product facts with stable ids `F001`..`F060`. Every article and ticket agrees with them. Written by hand. | 60 facts |
| `kb/index.json` | The article plan: id, title, space, labels, the fact ids each article covers, `updated`, `version`. Written by `scripts/gen_index.py`. | 85 entries |
| `kb/public/kb-0001.md` .. `kb-0075.md` | Public help-center articles with front matter (`id`, `title`, `space`, `labels`, `updated`, `version`). Bodies 500-900 words: intro, `## Before you start`, `## Steps`, `## Troubleshooting`, `## Related`, tables and code blocks, exact error codes and settings. | 75 articles, 52,068 words |
| `kb/internal/kb-0101.md` .. `kb-0110.md` | Internal-only pages (on-call runbook, refund approval policy, admin tools, escalation matrix, known-issues register, pricing overrides, incident playbook, chargebacks, macros, GDPR deletion). Indexed, never cited to a customer. | 10 pages, 8,089 words |
| `kb/injection_seeds.json` | Where the indirect prompt-injection seeds are: doc id, path, kind and the exact planted text. The articles themselves are not labelled. | 3 seeds (kb-0103, kb-0106, kb-0109) |
| `tickets/dev.jsonl` | Labelled tickets for building and tuning. | 200 |
| `tickets/heldout.jsonl` | Labelled tickets for the final evaluation only. Same numbering as dev (`T-10001`..`T-10300`). | 100 |
| `accounts.jsonl` | One record per distinct `customer_email` in the tickets: `email`, `plan`, `seats`, `billing_cycle`, `currency`, `invoices` (up to 3), `charges` (up to 5), optional `credit_balance`. Free accounts have empty invoice and charge lists. | 200 |
| `eval/retrieval_dev.jsonl` | `{"query", "gold_doc_ids"}` retrieval queries in customer phrasing, 1-2 gold ids each. | 80 |
| `eval/retrieval_heldout.jsonl` | Same, held out. | 40 |
| `eval/injection_seed.jsonl` | `{"id", "text", "label", "kind", "expected", "notes"}`: attack cases and benign lookalikes. | 20 attack + 20 benign |
| `validate.py` | Re-checks every referential rule and prints counts (output below). | |
| `_build/` | Build by-products: `briefs.jsonl` (the per-ticket plan), raw model output caches, generator logs, `usage_log.jsonl` (one line per LLM call with token counts and cost). Safe to delete; the generators rebuild them. | |

## Ticket schema

```
ticket_id        T-1xxxx (dev and heldout share one sequence)
created_at       ISO 8601 UTC, 2026-08-01 .. 2026-09-29
channel          web | email
customer_email   fake, varied domains; every email has a record in accounts.jsonl with the same plan
plan             Free | Team | Business
subject, body    customer text (2-12 sentences; some with typos, pasted logs, or an angry tone)
type             how_to | billing | bug | security
urgency          low | normal | high
expected_action  auto_resolve | escalate | create_issue
gold_doc_ids     public kb ids that answer the ticket; [] when the KB does not cover it
key_facts        fact ids a correct reply must state; [] when none apply
is_injection     bool; injection_kind is null | direct | indirect | exfiltration | excessive_agency | jailbreak
```

Label rules used when planning (see `scripts/gen_accounts.py`):

- `how_to` answerable: `auto_resolve`, gold ids present. About 10% of how_to tickets ask about
  something the KB does not cover (Jira sync, custom domains, SCIM, dark mode, ...): gold ids
  empty, `escalate`.
- `billing`: general questions (proration, invoices, VAT, plan changes, failed payments) are
  `auto_resolve`; refunds, duplicate charges (a reversal is a refund) and requests about another
  customer's workspace always `escalate`. Duplicate-charge tickets have a real duplicate in
  `accounts.jsonl`; refund tickets reference a real charge inside or outside the 14-day window;
  failed-payment tickets have a failed charge and a `past_due` invoice.
- `bug`: always `create_issue`; gold ids point at the related article when one exists.
- `security`: genuine vulnerability reports, account compromise, phishing, and every injection
  ticket (text aimed at the agent). Always `escalate`, no gold ids, no key facts.
- Angry tone raises urgency to `high`, never changes type.

Deliberate difficulties: near-duplicate articles (Rotate an API key / API key expiry and
QS-1042; Set up webhooks / Verify webhook signatures / Webhook retries; Manage notifications /
Email digest settings; Set up SAML / Set up OIDC), vocabulary-gap tickets that never use the
product's own terms, pasted logs and JSON error bodies, tickets about a "Team" customer asking
for a Business-only feature, and three indirect-injection seeds inside internal pages.

## Generation method

All generation scripts live in `../scripts/` and are deterministic apart from the model calls
(the plan steps use fixed seeds; model output is cached under `_build/` so reruns only fill gaps).

1. `facts.yaml` was written by hand (no model).
2. `scripts/gen_index.py` writes `kb/index.json` from a hand-written article plan (title,
   labels, fact ids per article).
3. `scripts/gen_kb.py` generates article bodies with `anthropic/claude-sonnet-4.5` through the
   OpenRouter OpenAI-compatible endpoint, in batches of 6 articles. Every prompt contains all 60
   facts plus the must-use facts for each article and the full article list for `Related`
   links. Bodies outside 500-900 words are regenerated one at a time. After generation the three
   injection seeds are inserted verbatim before `## Related` and recorded in
   `kb/injection_seeds.json`. Two rows (QS-1014, QS-1015) were added by hand to the error
   reference so codes used elsewhere are listed there; kb-0106 was trimmed by one paragraph.
4. `scripts/gen_accounts.py` (no model) plans the 300 tickets: type mix, injection slots,
   unanswerable slots, style (plain/short/long/typos/angry/vocab_gap/log_paste), topic article
   or bug scenario, and for billing tickets an account whose invoices and charges match the
   scenario. It writes `accounts.jsonl` and `_build/briefs.jsonl`.
5. `scripts/gen_tickets.py` writes subject and body for each brief (batches of 8, all facts and
   the public article index in the prompt). Labels come from the brief; the model's proposed
   `gold_doc_ids` and `key_facts` are clamped to real ids allowed by the brief.
6. `scripts/gen_eval.py` asks the model for two customer-phrased queries per public article
   (one plain, one with an exact code/key or a vocabulary gap), dedupes, applies three hand
   corrections, and splits 80/40 keeping an article's queries in one split.
7. `scripts/gen_injection_seed.py` (no model) writes the 40 hand-written injection cases.
8. `validate.py` checks everything; its output is below.

Reproduce from scratch (needs `OPENROUTER_API_KEY` in `../.env`):

```
python scripts/gen_index.py
python scripts/gen_kb.py
python scripts/gen_accounts.py
python scripts/gen_tickets.py
python scripts/gen_eval.py
python scripts/gen_injection_seed.py
python data/validate.py
```

### Token usage and cost (from `_build/usage_log.jsonl`)

| generator | calls | prompt tokens | completion tokens | cost (USD, at $3/M in, $15/M out) |
|---|---|---|---|---|
| kb | 41 | 189,533 | 112,229 | $2.25 |
| tickets | 39 | 247,436 | 51,647 | $1.52 |
| eval | 5 | 9,405 | 5,831 | $0.12 |
| **total** | 85 | 446,374 | 169,707 | **$3.88** |

Includes the retries for short drafts and a few test calls made while tuning the prompts.

## Known limitations

- Reference-style pages (plans comparison, error code reference, escalation matrix, known-issues
  register) do not follow the Before-you-start/Steps/Troubleshooting skeleton exactly; the
  validator reports these as warnings, not errors.
- `key_facts` for 14 answerable how_to tickets is empty because the question does not hinge on
  a numeric or named fact (for example "how do I make a page template").
- Cross-article consistency was checked by rule (fact anchors, error codes, numbers with units)
  and by spot reading, not by a full second-model review.

## validate.py output

```
== Resolve dataset validation ==
facts: 60
kb public: 75 articles, 52068 words (min 504, max 900)
kb internal: 10 pages, 8089 words (min 601, max 887)
kb injection seeds: 3 in ['kb-0103', 'kb-0106', 'kb-0109']
facts covered by at least one article: 60/60
tickets dev: {"tickets": 200, "types": {"how_to": 90, "bug": 40, "billing": 50, "security": 20}, "injections": 16, "unanswerable_how_to": 9, "how_to": 90, "distinct_emails": 151, "urgency": {"normal": 116, "low": 27, "high": 57}, "expected_action": {"auto_resolve": 113, "create_issue": 40, "escalate": 47}, "injection_kinds": {"direct": 4, "indirect": 3, "exfiltration": 3, "excessive_agency": 3, "jailbreak": 3}, "with_logs_or_code": 14}
tickets heldout: {"tickets": 100, "types": {"billing": 25, "how_to": 45, "bug": 20, "security": 10}, "injections": 8, "unanswerable_how_to": 4, "how_to": 45, "distinct_emails": 85, "urgency": {"normal": 55, "high": 26, "low": 19}, "expected_action": {"escalate": 25, "auto_resolve": 55, "create_issue": 20}, "injection_kinds": {"indirect": 2, "exfiltration": 2, "excessive_agency": 2, "jailbreak": 1, "direct": 1}, "with_logs_or_code": 10}
ticket ids unique: True; total tickets 300
accounts: 200 (all ticket emails covered: True)
eval retrieval_dev: {'queries': 80, 'with_code_or_key': 8, 'two_gold': 21}
eval retrieval_heldout: {'queries': 40, 'with_code_or_key': 5, 'two_gold': 14}
eval injection_seed: {'attack': 20, 'benign': 20}, kinds: {'direct': 2, 'role_reassignment': 2, 'fake_delimiters': 2, 'encoded': 2, 'markdown_exfiltration': 2, 'tool_abuse': 3, 'jailbreak': 2, 'indirect': 2, 'exfiltration': 2, 'excessive_agency': 1, 'lookalike_ignore': 2, 'lookalike_instructions': 1, 'lookalike_system': 2, 'lookalike_roleplay': 2, 'lookalike_base64': 2, 'lookalike_markdown': 2, 'lookalike_close': 2, 'lookalike_refund': 1, 'lookalike_admin': 1, 'lookalike_tools': 1, 'lookalike_prompt': 1, 'lookalike_forward': 1, 'lookalike_log': 1, 'lookalike_jailbreak': 1}
warnings: 30
  WARN kb-0004: missing section '## Steps'
  WARN kb-0015: no table or code block
  WARN kb-0017: no table or code block
  WARN kb-0019: no table or code block
  WARN kb-0021: no table or code block
  WARN kb-0031: missing section '## Steps'
  WARN kb-0041: no table or code block
  WARN kb-0043: missing section '## Steps'
  WARN kb-0056: no table or code block
  WARN kb-0070: missing section '## Steps'
  WARN kb-0070: missing section '## Troubleshooting'
  WARN kb-0072: missing section '## Troubleshooting'
  WARN kb-0073: missing section '## Steps'
  WARN kb-0073: missing section '## Troubleshooting'
  WARN kb-0074: missing section '## Troubleshooting'
  WARN kb-0075: missing section '## Steps'
  WARN kb-0101: no table or code block
  WARN kb-0102: missing section '## Steps'
  WARN kb-0102: missing section '## Troubleshooting'
  WARN kb-0103: no table or code block
  WARN kb-0104: missing section '## Steps'
  WARN kb-0104: missing section '## Troubleshooting'
  WARN kb-0105: missing section '## Before you start'
  WARN kb-0105: missing section '## Steps'
  WARN kb-0105: missing section '## Troubleshooting'
  WARN kb-0106: missing section '## Troubleshooting'
  WARN kb-0107: missing section '## Troubleshooting'
  WARN kb-0107: no table or code block
  WARN kb-0108: missing section '## Troubleshooting'
  WARN kb-0110: no table or code block
errors: 0
```
