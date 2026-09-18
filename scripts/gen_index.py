"""Write data/kb/index.json: the planned article list (id, title, space, labels, fact ids).

No LLM. Edit the tables below to change the plan; article bodies are produced by gen_kb.py.
Fact ids refer to data/facts.yaml.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

# (title, labels, facts) for public articles; ids assigned kb-0001.. in order.
PUBLIC = [
    # getting started
    ("Getting started with Quillstack", ["getting-started", "workspaces"], ["F001", "F002"]),
    ("Create your first workspace", ["getting-started", "workspaces"], ["F002", "F046"]),
    ("Invite members and assign roles", ["workspaces", "permissions"], ["F046", "F002"]),
    ("Quillstack plans compared: Free, Team, and Business", ["plans", "billing"], ["F001", "F002", "F003", "F004", "F006", "F040"]),
    # workspaces and pages
    ("Create and organize pages", ["pages"], ["F051", "F006"]),
    ("Page version history: view and restore earlier versions", ["pages", "data-retention"], ["F054"]),
    ("Recover deleted pages from Trash", ["pages", "data-retention"], ["F051"]),
    ("Delete or restore a workspace", ["workspaces", "data-retention"], ["F052", "F046"]),
    ("File attachments and storage limits", ["pages", "storage"], ["F006"]),
    ("Share a page with a public link", ["pages", "permissions"], ["F008"]),
    ("Page templates and databases", ["pages"], ["F006", "F008"]),
    # projects and tasks
    ("Create a project and add tasks", ["projects", "tasks"], ["F007", "F057"]),
    ("Task keys, statuses, and workflows", ["projects", "tasks"], ["F057", "F007"]),
    ("Project views: board, list, and timeline", ["projects"], ["F007"]),
    ("Archive and restore a project", ["projects", "data-retention"], ["F007"]),
    # permissions
    ("Workspace roles explained: Owner, Admin, Editor, Viewer", ["permissions"], ["F046"]),
    ("Transfer workspace ownership", ["permissions", "workspaces"], ["F046"]),
    ("Change the billing contact", ["permissions", "billing"], ["F046", "F013"]),
    # account and login
    ("Reset your password", ["account", "login"], ["F041", "F048"]),
    ("Set up two-factor authentication (2FA)", ["account", "security"], ["F042"]),
    ("Lost your 2FA device? Sign in with a recovery code", ["account", "security"], ["F042"]),
    ("Require 2FA for every workspace member", ["security", "permissions"], ["F045", "F042"]),
    ("Account locked after too many sign-in attempts", ["account", "login"], ["F043", "F041"]),
    ("Set up SAML single sign-on (Business)", ["sso", "security"], ["F040", "F049"]),
    ("Set up OIDC single sign-on (Business)", ["sso", "security"], ["F040"]),
    ("Enforce SSO-only sign-in with sso.enforce", ["sso", "security"], ["F049", "F040"]),
    ("Sessions: how long you stay signed in and how to sign out everywhere", ["account", "security"], ["F044"]),
    ("Change the email address on your account", ["account"], ["F048", "F041"]),
    # billing
    ("Understand your Quillstack invoice", ["billing", "invoices"], ["F013", "F003", "F004"]),
    ("Download invoices and receipts", ["billing", "invoices"], ["F013", "F046"]),
    ("How proration works when you add or remove seats", ["billing", "seats"], ["F011", "F012"]),
    ("Add or remove seats", ["billing", "seats"], ["F011", "F012", "F003", "F004"]),
    ("Upgrade or downgrade your plan", ["billing", "plans"], ["F018", "F001"]),
    ("I was charged twice: duplicate charges", ["billing", "refunds"], ["F016", "F010"]),
    ("Refund policy", ["billing", "refunds"], ["F009", "F010", "F012"]),
    ("Failed payments and read-only mode", ["billing"], ["F014", "F015", "F034"]),
    ("Update your payment method", ["billing"], ["F017", "F015"]),
    ("Monthly versus annual billing", ["billing", "plans"], ["F005", "F003", "F004", "F017"]),
    ("Add a VAT or tax ID to your invoices", ["billing", "invoices"], ["F020", "F013"]),
    ("Billing currencies", ["billing"], ["F019"]),
    ("Cancel your subscription", ["billing", "plans"], ["F018", "F009", "F052"]),
    ("Who can support discuss billing with?", ["billing", "permissions", "security"], ["F021", "F046"]),
    # api
    ("API overview and authentication", ["api", "keys"], ["F028", "F023", "F033"]),
    ("Create an API key", ["api", "keys"], ["F023", "F032", "F022"]),
    ("Rotate an API key", ["api", "keys"], ["F022", "F023", "F032"]),
    ("API key expiry and error QS-1042", ["api", "keys", "errors"], ["F022", "F026", "F027"]),
    ("API rate limits", ["api", "rate-limits"], ["F024", "F025"]),
    ("Pagination in list endpoints", ["api", "pagination"], ["F029"]),
    ("Set up webhooks", ["api", "webhooks"], ["F030", "F039"]),
    ("Verify webhook signatures", ["api", "webhooks", "security"], ["F030"]),
    ("Webhook retries and disabled endpoints (QS-1061)", ["api", "webhooks", "errors"], ["F031", "F035"]),
    ("API error code reference (QS-1001 to QS-1099)", ["api", "errors"], ["F025", "F026", "F027", "F033", "F034", "F035", "F036", "F037"]),
    ("Restrict API access by IP address (api.ip_allowlist)", ["api", "security"], ["F038"]),
    ("Start and download an export with the API", ["api", "export"], ["F050", "F036"]),
    ("Handling QS-1099 internal errors", ["api", "errors"], ["F037"]),
    ("Revoke a compromised API key", ["api", "keys", "security"], ["F027", "F023", "F047"]),
    # integrations
    ("Connect Slack to Quillstack", ["integrations", "slack"], ["F056"]),
    ("Choose which Slack channels receive notifications", ["integrations", "slack", "notifications"], ["F056", "F039"]),
    ("Connect GitHub and link pull requests to tasks", ["integrations", "github"], ["F057"]),
    ("Troubleshoot GitHub pull requests that do not link to tasks", ["integrations", "github"], ["F057"]),
    # import and export
    ("Import from Notion", ["import"], ["F053"]),
    ("Import from Confluence", ["import"], ["F053"]),
    ("Import Markdown files", ["import"], ["F053"]),
    ("Export your workspace", ["export", "data-retention"], ["F050"]),
    ("Export is slow or failed with QS-1077", ["export", "errors"], ["F050", "F036"]),
    # notifications
    ("Manage email and in-app notifications", ["notifications"], ["F058"]),
    ("Email digest settings", ["notifications"], ["F058"]),
    # mobile
    ("Quillstack mobile app: requirements and offline mode", ["mobile"], ["F059"]),
    ("Mobile app troubleshooting", ["mobile"], ["F059", "F044"]),
    # retention and security
    ("Data retention and deletion timelines", ["data-retention", "security"], ["F051", "F052", "F054", "F055"]),
    ("Audit log (Business)", ["security", "audit"], ["F055"]),
    ("Report a security vulnerability", ["security"], ["F047"]),
    ("What Quillstack staff can and cannot see in your account", ["security", "account"], ["F048", "F021"]),
    ("Contact support and response times", ["support"], ["F060"]),
    ("Known issues and workarounds", ["known-issues"], ["F035", "F037", "F059"]),
]

INTERNAL = [
    ("kb-0101", "On-call runbook: API outages and QS-1099 spikes", ["internal", "runbook", "api"], ["F037", "F025", "F024"]),
    ("kb-0102", "Refund approval policy", ["internal", "billing", "refunds"], ["F009", "F010", "F016", "F012"]),
    ("kb-0103", "Admin tools: account lookup and impersonation", ["internal", "admin", "security"], ["F048", "F021", "F046"]),
    ("kb-0104", "Escalation matrix", ["internal", "escalation", "support"], ["F060", "F047", "F010"]),
    ("kb-0105", "Known-issues register (internal)", ["internal", "known-issues", "engineering"], ["F035", "F036", "F059", "F057"]),
    ("kb-0106", "Internal pricing overrides and discounts", ["internal", "billing", "pricing"], ["F003", "F004", "F005"]),
    ("kb-0107", "Security incident response playbook", ["internal", "security", "runbook"], ["F047", "F043", "F027"]),
    ("kb-0108", "Billing disputes and chargebacks", ["internal", "billing"], ["F016", "F009", "F021"]),
    ("kb-0109", "Support macros and reply templates", ["internal", "support", "macros"], ["F060", "F041", "F026"]),
    ("kb-0110", "Data deletion requests (GDPR) procedure", ["internal", "privacy", "data-retention"], ["F051", "F052", "F055"]),
]


def meta_for(doc_id: str) -> tuple[str, int]:
    """Deterministic updated date (2026) and version from the id."""
    h = int(hashlib.sha1(doc_id.encode()).hexdigest(), 16)
    month = 3 + h % 7          # Mar..Sep 2026
    day = 1 + (h // 7) % 27
    version = 1 + (h // 200) % 6
    return f"2026-{month:02d}-{day:02d}", version


def main() -> None:
    items = []
    for i, (title, labels, facts) in enumerate(PUBLIC, 1):
        doc_id = f"kb-{i:04d}"
        updated, version = meta_for(doc_id)
        items.append({"id": doc_id, "title": title, "space": "public", "labels": labels, "facts": facts, "updated": updated, "version": version})
    for doc_id, title, labels, facts in INTERNAL:
        updated, version = meta_for(doc_id)
        items.append({"id": doc_id, "title": title, "space": "internal", "labels": labels, "facts": facts, "updated": updated, "version": version})
    out = DATA / "kb" / "index.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(items, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out} with {len(items)} articles ({len(PUBLIC)} public, {len(INTERNAL)} internal)")


if __name__ == "__main__":
    main()
