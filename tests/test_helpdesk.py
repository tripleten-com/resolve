"""Tests for the provided helpdesk app, against a temp DB seeded from fixtures."""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpdesk.app import app  # noqa: E402


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("HELPDESK_DB", str(tmp_path / "test.db"))
    monkeypatch.setenv("HELPDESK_AUDIT_LOG", str(tmp_path / "audit.log"))
    monkeypatch.setenv("HELPDESK_DATA_DIR", str(tmp_path / "no-data"))  # force fixtures
    with TestClient(app) as c:
        yield c


def test_list_tickets_newest_first_without_bodies(client):
    r = client.get("/tickets")
    assert r.status_code == 200
    tickets = r.json()
    assert [t["ticket_id"] for t in tickets] == ["T-10003", "T-10002", "T-10001"]
    assert "body" not in tickets[0]
    assert client.get("/tickets", params={"limit": 1}).json()[0]["ticket_id"] == "T-10003"
    assert client.get("/tickets", params={"status": "resolved"}).json() == []


def test_get_ticket_with_comments(client):
    r = client.get("/tickets/T-10001")
    assert r.status_code == 200
    body = r.json()
    assert body["subject"] == "How do I rotate an API key?"
    assert body["status"] == "open"
    assert body["comments"] == []
    assert client.get("/tickets/T-99999").status_code == 404


def test_add_comment(client, tmp_path):
    r = client.post("/tickets/T-10001/comments",
                    json={"body": "Rotate it under Settings > API keys.", "author": "resolve", "internal": False})
    assert r.status_code == 201
    comment = r.json()
    assert comment["ticket_id"] == "T-10001" and comment["internal"] is False
    assert client.get("/tickets/T-10001").json()["comments"][0]["body"] == comment["body"]
    assert "POST /tickets/T-10001/comments" in (tmp_path / "audit.log").read_text()


def test_status_change_valid_and_invalid(client):
    r = client.patch("/tickets/T-10002/status", json={"status": "escalated"})
    assert r.status_code == 200 and r.json()["status"] == "escalated"
    assert client.get("/tickets/T-10002").json()["status"] == "escalated"
    assert client.patch("/tickets/T-10002/status", json={"status": "closed"}).status_code == 422


def test_issue_search(client):
    hits = client.get("/issues", params={"q": "QS-1042"}).json()
    assert [i["issue_id"] for i in hits] == ["ISS-1001"]
    assert len(client.get("/issues").json()) == 8


def test_create_issue_twice_with_same_key_returns_one_issue(client):
    payload = {"title": "QS-1042 on a 30-day-old key", "body": "see T-10003",
               "ticket_id": "T-10003", "idempotency_key": "T-10003:create_issue:1a2b3c4d"}
    first = client.post("/issues", json=payload)
    second = client.post("/issues", json=payload)
    assert first.status_code == 201 and first.json()["created"] is True
    assert second.status_code == 200 and second.json()["created"] is False
    assert first.json()["issue_id"] == second.json()["issue_id"] == "ISS-1009"
    assert len(client.get("/issues", params={"q": "30-day-old"}).json()) == 1


def test_account_lookup_and_404(client):
    r = client.get("/accounts/ops@brightfield.io")
    assert r.status_code == 200
    assert r.json()["plan"] == "Team" and r.json()["seats"] == 14
    assert isinstance(r.json()["invoices"], list) and isinstance(r.json()["charges"], list)
    assert client.get("/accounts/nobody@example.com").status_code == 404
