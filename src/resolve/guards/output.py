"""The output guard: secret scrub, citation check, link check, schema check, write flag.

Every flag goes to the state; `route` never auto-resolves a flagged draft.
"""
from __future__ import annotations

import re
from urllib.parse import urlparse

from resolve.config import guards_config
from resolve.guards import guards_enabled
from resolve.guards.input import scrub_secrets

MD_IMAGE = re.compile(r"!\[[^\]]*\]\(\s*(?P<url>[^)\s]+)[^)]*\)")
MD_LINK = re.compile(r"\[(?P<text>[^\]]*)\]\(\s*(?P<url>https?://[^)\s]+)[^)]*\)")
BARE_URL = re.compile(r"(?<![\(\[])https?://[^\s<>\)\]]+")
DOC_ID = re.compile(r"\bkb-\d{4}\b")


def _host_allowed(url: str, allowed: list[str]) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return any(host == a or host.endswith("." + a) for a in allowed)


def strip_links(draft: str, allowed: list[str]) -> tuple[str, list[str]]:
    """Remove every link or image whose host is not on the allowed list; flag each host once."""
    flags: list[str] = []

    def _flag(url: str) -> None:
        host = (urlparse(url).hostname or "?").lower()
        f = f"link_stripped:{host}"
        if f not in flags:
            flags.append(f)

    def img(m: re.Match) -> str:
        if _host_allowed(m.group("url"), allowed):
            return m.group(0)
        _flag(m.group("url"))
        return ""

    def link(m: re.Match) -> str:
        if _host_allowed(m.group("url"), allowed):
            return m.group(0)
        _flag(m.group("url"))
        return m.group("text")

    def bare(m: re.Match) -> str:
        if _host_allowed(m.group(0), allowed):
            return m.group(0)
        _flag(m.group(0))
        return "[link removed]"

    draft = MD_IMAGE.sub(img, draft)
    draft = MD_LINK.sub(link, draft)
    draft = BARE_URL.sub(bare, draft)
    return draft, flags


def check_citations(draft: str, citations: list[str], docs: list[dict]) -> list[str]:
    """Every citation must be a doc_id in docs and none may be internal-only."""
    flags: list[str] = []
    by_id = {d.get("doc_id"): d for d in docs or []}
    mentioned = set(citations or []) | set(DOC_ID.findall(draft or ""))
    for cid in sorted(mentioned):
        if cid not in by_id:
            flags.append(f"citation_unknown:{cid}")
        elif by_id[cid].get("space") == "internal":
            flags.append(f"citation_internal:{cid}")
    return flags


def guard_draft(draft: str, citations: list[str], docs: list[dict], tool_calls: list[dict]) -> tuple[str, dict]:
    """Return (cleaned draft, guard_output). guard_output = flags, ok."""
    cfg = guards_config().get("output", {})
    if not guards_enabled():
        return draft, {"flags": ["guards_disabled"], "ok": True}

    flags: list[str] = []
    draft, secret_flags = scrub_secrets(draft or "")
    flags += [f"secret_scrubbed:{f.split(':', 1)[1]}" for f in secret_flags]

    flags += check_citations(draft, citations, docs)

    draft, link_flags = strip_links(draft, list(cfg.get("allowed_link_hosts", [])))
    flags += link_flags

    n = len(draft.strip())
    if n < int(cfg.get("min_draft_chars", 40)):
        flags.append("schema:draft_too_short")
    if n > int(cfg.get("max_draft_chars", 4000)):
        flags.append("schema:draft_too_long")
    if not citations:
        flags.append("schema:no_citations")

    for call in tool_calls or []:
        if call.get("write") and call.get("node") not in ("resolve", "escalate"):
            f = f"wrote:{call['name']}"
            if f not in flags:
                flags.append(f)
        if call.get("dropped"):
            flags.append(f"chunk_dropped:{call['dropped']}")

    return draft, {"flags": flags, "ok": not flags}
