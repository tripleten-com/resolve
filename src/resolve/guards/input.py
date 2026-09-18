"""The input guard: length cap, secret scrub, encoded-text flags, injection risk, quarantine.

It labels and quarantines. It never decides anything else.
"""
from __future__ import annotations

import re

from resolve.config import guards_config
from resolve.guards import guards_enabled
from resolve.guards.injection import encoded_flags, score_text, threshold

SECRET_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("api_key", re.compile(r"qs_live_[A-Za-z0-9]{6,}")),
    ("webhook_secret", re.compile(r"whsec_[A-Za-z0-9]{6,}")),
    ("bearer", re.compile(r"(?i)bearer\s+[A-Za-z0-9._\-]{16,}")),
    ("generic_key", re.compile(r"\b(?:sk|pk|rk|key|token|api)[-_](?:live|test|prod)?[-_]?[A-Za-z0-9]{20,}\b")),
    ("aws_key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("password", re.compile(r"(?i)\b(password|passwd|pwd|passphrase)(\s+is|\s*:|\s*=)\s*(\S{4,})")),
    ("card", re.compile(r"\b(?:\d[ -]?){13,16}\b")),
]


def _luhn(digits: str) -> bool:
    total, alt = 0, False
    for d in reversed(digits):
        n = int(d)
        if alt:
            n = n * 2 - 9 if n * 2 > 9 else n * 2
        total += n
        alt = not alt
    return total % 10 == 0


def scrub_secrets(text: str) -> tuple[str, list[str]]:
    """Replace anything shaped like a key, password, or card number with a marker."""
    flags: list[str] = []
    if not text:
        return text, flags

    def make_repl(kind: str):
        def _r(m: re.Match) -> str:
            if kind == "card" and not _luhn(re.sub(r"\D", "", m.group(0))):
                return m.group(0)
            flag = f"secret:{kind}"
            if flag not in flags:
                flags.append(flag)
            if kind == "password":
                return f"{m.group(1)}{m.group(2)} [REDACTED:password]"
            return f"[REDACTED:{kind}]"
        return _r

    for kind, pat in SECRET_PATTERNS:
        text = pat.sub(make_repl(kind), text)
    return text, flags


def guard_ticket(ticket: dict) -> tuple[dict, dict]:
    """Return (cleaned ticket, guard_input). guard_input = flags, risk, quarantined."""
    cfg = guards_config().get("input", {})
    max_chars = int(cfg.get("max_chars", 6000))
    ticket = dict(ticket)
    subject, body = ticket.get("subject") or "", ticket.get("body") or ""
    flags: list[str] = []

    if not guards_enabled():
        return ticket, {"flags": ["guards_disabled"], "risk": 0.0, "quarantined": False}

    if len(body) > max_chars:
        body = body[:max_chars] + "\n[truncated by input_guard]"
        flags.append("truncated")

    subject, f1 = scrub_secrets(subject)
    body, f2 = scrub_secrets(body)
    flags += [f for f in f1 + f2 if f not in flags]

    text = f"{subject}\n{body}"
    flags += encoded_flags(text)

    s = score_text(text)
    flags += [f"injection:{h}" for h in s["heuristics"]]
    if s["classifier"] is not None:
        flags.append(f"classifier:{s['classifier']:.2f}")
    quarantined = s["risk"] >= threshold()
    if quarantined:
        flags.append("quarantined")

    ticket["subject"], ticket["body"] = subject, body
    return ticket, {"flags": flags, "risk": s["risk"], "quarantined": quarantined}
