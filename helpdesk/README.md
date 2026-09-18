# Quillstack helpdesk (provided)

A small FastAPI + SQLite app that stands in for three internal systems: the
ticketing system (tickets and comments), the bug tracker (issues), and the
customer-account lookup. You do not modify it. In Task 3 you write an MCP
server that calls its HTTP API.

## Run it locally

From the repo root:

```bash
uvicorn helpdesk.app:app --port 8100
```

On first start the app creates `helpdesk/helpdesk.db` and seeds it from
`data/tickets/dev.jsonl`, `data/tickets/heldout.jsonl` (ticket fields only, no
labels) and `data/accounts.jsonl`. If those files are missing it seeds three
tickets and one account from `helpdesk/fixtures.jsonl` instead. Eight bug
issues are always seeded so `GET /issues?q=` has something to find.

To reseed by hand:

```bash
python -m helpdesk.seed
```

Or in Compose (see `compose.snippet.yaml`): `docker compose up helpdesk`, then
the API is on `http://localhost:8100`.

Environment variables: `HELPDESK_DB` (SQLite path, default
`helpdesk/helpdesk.db`), `HELPDESK_AUDIT_LOG` (default `helpdesk/audit.log`),
`HELPDESK_DATA_DIR` (default `data/` next to this folder).

## Endpoints

All requests and responses are JSON.

| Method | Path | Body / query | Returns |
|---|---|---|---|
| GET | `/health` | | `{"status": "ok", "tickets": n}` |
| GET | `/tickets` | `?status=open&limit=50` | list, newest first, no bodies |
| GET | `/tickets/{ticket_id}` | | ticket with body and `comments` |
| POST | `/tickets/{ticket_id}/comments` | `{"body", "author", "internal"}` | 201, the comment |
| PATCH | `/tickets/{ticket_id}/status` | `{"status"}` one of `open`, `pending_customer`, `escalated`, `resolved` | the ticket summary; 422 for any other value |
| GET | `/issues` | `?q=text` substring match on title and body, max 20 | list of issues |
| POST | `/issues` | `{"title", "body", "ticket_id", "idempotency_key"?}` | 201 with `"created": true`, or 200 with `"created": false` and the existing issue when the key was seen before |
| GET | `/accounts/{email}` | | `plan`, `seats`, `invoices`, `charges`; 404 if unknown |
| POST | `/admin/reset` | | drops and reseeds the database |

Unknown ticket ids return 404. Interactive docs are at `/docs`.

## Audit log

Every write (`POST /tickets/{id}/comments`, `PATCH /tickets/{id}/status`,
`POST /issues`, `POST /admin/reset`) appends one line to `helpdesk/audit.log`:
timestamp, method, path, and a short summary of the body. Use it to show what
your agent actually did, and to prove that a replayed write with the same
idempotency key created one issue, not two.

```
2026-09-17T10:02:41+00:00 POST /issues {"issue_id": "ISS-1009", "created": true, "ticket_id": "T-10003", ...}
2026-09-17T10:02:42+00:00 POST /issues {"issue_id": "ISS-1009", "created": false, "idempotency_key": "T-10003:create_issue:1a2b3c4d"}
```

## Start fresh

`POST /admin/reset` drops every table and reseeds from the data files. Run it
before an evaluation so earlier comments, status changes and issues do not
leak into the next run:

```bash
curl -X POST http://localhost:8100/admin/reset
```

## Tests

```bash
python -m pytest tests/test_helpdesk.py
```
