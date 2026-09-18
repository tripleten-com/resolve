import hashlib
import json

from resolve.tools.idempotency import idempotency_key


def test_key_shape_and_hash():
    args = {"title": "QS-1077 export fails", "body": "steps", "ticket_id": "T-10004"}
    key = idempotency_key("T-10004", "create_issue", args)
    expected = "T-10004:create_issue:" + hashlib.sha256(json.dumps(args, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:8]
    assert key == expected
    assert len(key.split(":")[-1]) == 8


def test_key_ignores_argument_order():
    a = idempotency_key("T-1", "create_issue", {"title": "x", "body": "y"})
    b = idempotency_key("T-1", "create_issue", {"body": "y", "title": "x"})
    assert a == b


def test_key_changes_with_args_tool_or_ticket():
    base = idempotency_key("T-1", "create_issue", {"title": "x"})
    assert idempotency_key("T-1", "create_issue", {"title": "x2"}) != base
    assert idempotency_key("T-1", "add_ticket_comment", {"title": "x"}) != base
    assert idempotency_key("T-2", "create_issue", {"title": "x"}) != base
