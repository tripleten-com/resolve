"""The idempotency key every write carries: f"{ticket_id}:{tool}:{sha256(json.dumps(args, sort_keys=True))[:8]}"."""
from __future__ import annotations

import hashlib
import json


def idempotency_key(ticket_id: str, tool: str, args: dict) -> str:
    payload = json.dumps(args, sort_keys=True, ensure_ascii=False, default=str)
    return f"{ticket_id}:{tool}:{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:8]}"
