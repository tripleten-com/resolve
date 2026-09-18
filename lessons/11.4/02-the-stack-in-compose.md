The service from the last lesson runs under uvicorn in your virtual environment, against an OpenSearch and a helpdesk that you started by hand. That is fine for you. It is not something Priya's team, or the reviewer of your repository, can start. The Project 2 answer was one file and one command: declare every container in `compose.yaml` and bring the whole system up with `docker compose up -d`. This lesson adds the agent to the stack that already holds OpenSearch and the helpdesk, and then checks three things a reviewer will check: the key is not in the image, one ticket runs through the published port, and `/health` tells the truth when a dependency dies.

<callout icon="📖" color="gray_bg">
	The Compose mechanics here, service names as hostnames, configuration from the environment, health checks and health-gated startup, and named volumes, are the ones from Project 2. If `docker compose config` or `depends_on` with a condition looks unfamiliar, reread the Compose lesson there before this one.
</callout>

## The image

The **Dockerfile** describes the agent image, the filesystem the service runs in. Here it is, whole, because every line is doing something:

```docker
FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app/src \
    HF_HOME=/app/var/hf TOKENIZERS_PARALLELISM=false

COPY requirements.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

COPY src ./src
COPY configs ./configs
RUN mkdir -p /app/data /app/var

EXPOSE 8200
CMD ["uvicorn", "resolve.api:app", "--host", "0.0.0.0", "--port", "8200"]
```

`python:3.12-slim` is the small official base, the same one Project 2 used. `PYTHONPATH=/app/src` replaces the `PYTHONPATH=src` you export on your laptop. `HF_HOME=/app/var/hf` tells the Hugging Face libraries where to cache downloaded models, and pointing it under `var/` is what lets the next section keep the injection classifier outside the container. Requirements are copied and installed before the source, so a code change does not repeat the install; `--extra-index-url` pulls the CPU build of torch, which keeps the image at about two gigabytes; the default wheels carry GPU libraries that would triple it. Then `src` and `configs` and nothing else: no `data/` (mounted at run time), no `.env`, no `var/`. `--host 0.0.0.0` makes uvicorn listen on the container's network interface, since the default `127.0.0.1` would be reachable only from inside the container.

The one thing missing from that file is any secret. There is no `ENV LLM_API_KEY`, no `COPY .env`. The image is the code and its libraries, and it is the same image whichever key, model, or provider runs it.

## The agent service

The `agent` stanza joins `opensearch` and `helpdesk` in `compose.yaml`:

```yaml
  agent:
    build: .
    container_name: resolve-agent
    ports:
      - "${AGENT_PORT:-8200}:8200"
    environment:
      - LLM_BASE_URL=${LLM_BASE_URL}
      - LLM_API_KEY=${LLM_API_KEY}
      - LLM_MODEL=${LLM_MODEL}
      - LLM_MODEL_REVIEW=${LLM_MODEL_REVIEW:-${LLM_MODEL}}
      - LANGSMITH_TRACING=${LANGSMITH_TRACING:-false}
      - LANGSMITH_API_KEY=${LANGSMITH_API_KEY:-}
      - LANGSMITH_PROJECT=${LANGSMITH_PROJECT:-resolve}
      - OPENSEARCH_URL=http://opensearch:9200
      - HELPDESK_URL=http://helpdesk:8100
    volumes:
      - ./data:/app/data:ro
      - resolve_agent:/app/var
    depends_on:
      opensearch:
        condition: service_healthy
      helpdesk:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8200/health')"]
      interval: 30s
      timeout: 15s
      retries: 5
      start_period: 90s
```

Take the blocks in turn.

`build: .` points at the Dockerfile in the repository root, the way `build: ./helpdesk` points at the helpdesk's. The published port comes from `AGENT_PORT` with 8200 as the default, so two people on one machine can run the stack without editing the file.

The `environment` block is where the `.env` file you have used all project meets Compose. `${LLM_API_KEY}` is a **variable reference**: at `docker compose up`, Compose reads `.env` from the directory holding `compose.yaml` and substitutes the value into the container's environment. The value passes from your file to the running process and never touches the image. `${LLM_MODEL_REVIEW:-${LLM_MODEL}}` is the same syntax with a default: if the reviewer model is unset, use the drafting model.

The last two lines override two values from `.env`. On your laptop `OPENSEARCH_URL` is `http://localhost:9200` and `HELPDESK_URL` uses `HELPDESK_PORT`. Inside the Compose network, `opensearch` and `helpdesk` are hostnames, and the helpdesk listens on 8100 inside its container whatever port it publishes. Nothing in `src/` knows which of the two it is running in; that is the payoff from the last lesson's "everything from the environment".

`./data:/app/data:ro` bind-mounts the knowledge base, tickets, and facts read-only, since the image does not carry them and the service should never write there. `resolve_agent:/app/var` is a **named volume**, storage that outlives any container. It holds two things: `checkpoints.db`, so a run paused for approval survives a container restart, and `hf/`, the model cache. The injection classifier is about 740 MB and the embedding model adds about 90 MB. With the cache on a volume they download on the first ticket after the first `up`, and never again, through every rebuild of the image.

`depends_on` with `condition: service_healthy` is the difference between started and ready. The agent's `build_graph` starts the helpdesk MCP server, and its first ticket needs the index, so the agent must not start until both dependencies pass their own health checks. Its own check is the one-line Python request from Project 2 (the slim image has no curl), against the `/health` route from the last lesson, with a `start_period` of ninety seconds so the graph build and a possible model download do not count as failures.

## Proving there is no key in the image

The task page asks you to read the resolved file. Three commands answer the question properly.

```bash
grep -n LLM_API_KEY compose.yaml Dockerfile
docker compose config --no-interpolate | grep -A3 LLM_API_KEY
docker image inspect resolve-agent --format '{{join .Config.Env "\n"}}'
```

```plain text
compose.yaml:53:      - LLM_API_KEY=${LLM_API_KEY}
      - LLM_API_KEY=${LLM_API_KEY}
      - LLM_MODEL=${LLM_MODEL}
      - LLM_MODEL_REVIEW=${LLM_MODEL_REVIEW:-${LLM_MODEL}}
PATH=/usr/local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
LANG=C.UTF-8
PYTHON_VERSION=3.12.14
PYTHONDONTWRITEBYTECODE=1
PYTHONUNBUFFERED=1
PYTHONPATH=/app/src
HF_HOME=/app/var/hf
TOKENIZERS_PARALLELISM=false
```

The first line shows the key appears in exactly one place, as a variable. The second is `docker compose config`, the parser's view of the file, with `--no-interpolate` so it prints the references rather than their values. The third lists every environment variable baked into the image, and no key is among them. One warning about the second command: plain `docker compose config`, without the flag, substitutes your `.env` values into what it prints, including the key. It is a good way to check that a default resolved the way you meant, and a bad thing to paste into a report, a ticket, or a chat window.

## Bring it up

```bash
docker compose up -d --build
docker compose ps
```

```plain text
SERVICE      STATUS                       PORTS
agent        Up 2 minutes (healthy)       0.0.0.0:8200->8200/tcp
helpdesk     Up 2 minutes (healthy)       0.0.0.0:8110->8100/tcp
opensearch   Up 2 minutes (healthy)       0.0.0.0:9200->9200/tcp
```

The first build takes a few minutes, most of it the torch wheel; rebuilds after a code change take seconds because the install layer is cached. The agent shows `(healthy)` only after its `/health` passed, which means the graph built inside the container, the MCP server started, and all three dependencies answered. If the helpdesk is fresh, seed it with `curl -s -X POST localhost:${HELPDESK_PORT}/admin/reset` as the README describes, and index the knowledge base from your laptop as before; the index lives in OpenSearch, so the container sees it.

Then one ticket through the published port. T-10118 is a dev how-to, "New user - where to start?", open with no comments:

```bash
curl -s -w '\nround trip: %{time_total}s\n' -X POST localhost:8200/tickets/T-10118/run
```

```json
{"ticket_id": "T-10118", "paused": true,
 "pending_write": {"tool": "add_ticket_comment", "args": {"ticket_id": "T-10118", "body": "Resolve escalated this ticket.\nReason: review verdict is revise (...)", "internal": true},
                   "idempotency_key": "T-10118:add_ticket_comment:0297d0d2", "node": "escalate", ...},
 "decision": null,
 "classification": {"type": "how_to", "urgency": "normal", "aimed_at_agent": false, "confidence": 0.95},
 "confidence": 0.95, "citations": ["kb-0001"],
 "review": {"verdict": "revise", "reasons": ["The draft mentions 'configure notifications under Settings > Notifications' but no retrieved chunk supports this claim ...", "..."],
            "unsupported_claims": ["configure notifications under Settings > Notifications", "..."], "revision_count": 2},
 "usage": {"input_tokens": 32426, "output_tokens": 2008, "cost_usd": 0.064466, "latency_ms": 43792},
 "http_ms": 43851}
round trip: 43.873055s
```

A different outcome from the last lesson, and a useful one. The reviewer sent the draft back for two unsupported claims, the specialist revised, the reviewer sent it back again, and the revision limit from `configs/graph.yaml` ended the loop: two revise verdicts means `escalate`. So the run paused in `escalate` on an internal note that carries the reasons, the tool calls, and the draft. Two approvals through `POST /tickets/T-10118/approve` post the note and set the status, and `curl localhost:${HELPDESK_PORT}/tickets/T-10118` shows `escalated` with one internal comment. A ticket that should have been easy cost more than the one that was auto-resolved, and the next lesson is about why.

The timing tells the same story as before. Round trip 43,873 ms, `http_ms` 43,851, `latency_ms` 43,792: the container's HTTP layer adds a few tens of milliseconds, and the first ticket after a fresh `up` adds the classifier load or download on top.

## Stop a dependency and watch

`depends_on` orders startup. It does nothing once the stack is running: if OpenSearch dies at noon, the agent container stays up, its process stays alive, and Docker's own status still says `Up`. The only thing that knows the difference is the health route.

```bash
docker compose stop opensearch
curl -s -w '\nHTTP %{http_code}\n' localhost:8200/health
```

```json
{"status":"degraded","checks":{"llm":"ok","opensearch":"ConnectionError: ConnectionError(HTTPConnection(host='opensearch', port=9200): Failed to resolve ","helpdesk":"ok","graph":"ok"}}
HTTP 503
```

The failing check is named, with the reason: the hostname `opensearch` no longer resolves because the container is gone from the network. The status code is 503, so the agent's own Compose health check starts failing too, and after `retries` it shows `(unhealthy)` in `docker compose ps`. A process supervisor or a load balancer reads exactly that. Then:

```bash
docker compose start opensearch
curl -s localhost:8200/health
```

```json
{"status":"ok","checks":{"llm":"ok","opensearch":"ok","helpdesk":"ok","graph":"ok"}}
```

About fifteen seconds later, once OpenSearch passes its own check, the agent is back to `ok` without a restart, because the graph holds no connection open across requests. That is the behaviour to show a reviewer: the service reports its dependencies honestly, in both directions.

<callout icon="🤖" color="blue_bg">
	**Working with AI**
	When the stack misbehaves, the useful thing to hand an assistant is all four sides at once: `docker compose ps`, `docker compose logs agent`, `docker compose logs helpdesk`, and the body of `/health`. The agent logging `ConnectError` is usually a symptom of the helpdesk or OpenSearch being unhealthy, and an assistant that sees only the agent's log will send you to the wrong file. Never paste the output of plain `docker compose config`; use `--no-interpolate`.
</callout>

The next lesson, **Cost, Latency, and Budgets**, reads the `usage` field these runs produced, shows where the tokens went node by node, and turns the numbers into the floors CI enforces.
