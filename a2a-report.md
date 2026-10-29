# EZStore A2A Application — Implementation Report

HTML version with the full page layout: [a2a-report.html](a2a-report.html).

**Task:** complete an A2A application using the A2A protocol (bonus).
**Scope:** the quotation agent hands an order request to the order agent over A2A, and the answer returns in the same quotation chat. No new service, no frontend change, no new MCP tool.

**Status:** implemented and demonstrated in the browser. The container is on the real model (`AGENT_MODEL=gemini-3.1-flash-lite`, `ORDER_AGENT_FAKE_LLM` empty). At capture time the 3.1 flash-lite endpoint was intermittently returning `503 UNAVAILABLE`; that is an external availability issue, not the A2A path. The zero-cost deterministic mode is available for a repeatable demo.

## 1. What the demo shows

The user stays in the normal quotation UI at `http://localhost:3000/agent`. There is no app switch and no special command.

### 1.1 Real-model single turn

Prompt:

```
我要查訂單，訂單編號：ORD-20260118-001
```

![Quotation chat: order lookup handed to the order agent over A2A](.screenshots/a2a-handoff.png)

The answer is produced by the order agent: order fields, items, a plain-text `http://localhost:3000/orders/1` link, and the footer:

```
（本回覆由訂單 Agent 透過 A2A 協定提供）
```

This screenshot was captured while `gemini-3.1-flash-lite` was available.

### 1.2 Multi-turn over A2A

Turn 1 asks for an order without a number; the order agent asks for the identifier. Turn 2 supplies it; the order agent continues from its A2A session state and returns the details.

![Turn 1: the order agent asks for the order number](.screenshots/a2a-multiturn-1.png)

![Turn 2: the order details after the number is supplied](.screenshots/a2a-multiturn-2.png)

These two screenshots were captured in the supported zero-cost mode (`ORDER_AGENT_FAKE_LLM=1`) because the 3.1 model endpoint was returning `503` at that moment. The A2A hop, the `context_id`, the MCP call and the UI are the same as the real-model path; only the order classifier is replaced by its rule-based equivalent. In this mode the whole two-turn exchange uses **0 Gemini calls**.

## 2. How it works

```
/agent  user message
  → quotation.guard                 deterministic, no model call
      ├─ order request  → order_handoff
      │                     │  A2A: card discovery + message/send
      │                     ▼
      │                  order_agent (same port, /a2a/order_agent)
      │                     ├─ classifier (1 call in real mode)
      │                     └─ MCP get_order / list_orders / order_summary
      │                     ▼
      └─ otherwise      → classify_intent → existing quotation flow
                              │
                              ▼
                           respond  (single user-facing text event)
```

Core implementation:

- **Server:** `attach_a2a_routes()` publishes an agent card and JSON-RPC endpoint for each app. The card URL is built from `A2A_BASE_URL` (default `http://localhost:8080`).
- **Client:** `call_remote_agent()` resolves `/.well-known/agent-card.json`, creates an A2A client, sends `message/send`, and returns the last non-JSON text from the stream.
- **Handoff:** `quotation_agent.agent.order_handoff` calls the order agent with the quotation session id as `context_id`, so the order agent keeps its clarification state on the next turn.
- **Boundary:** `routing.detect_order_request()` excludes quotation operations, especially `報價單轉訂單`, so the existing six quotation commands are unchanged.

## 3. A2A evidence

The order agent publishes a standard card:

```json
{
  "name": "order_agent",
  "description": "EZStore order inquiry agent (read-only): look up an order by number, list a customer's orders, or summarise order counts and totals.",
  "version": "0.0.0",
  "supportedInterfaces": [
    {
      "url": "http://localhost:8080/a2a/order_agent",
      "protocolBinding": "JSONRPC",
      "protocolVersion": "1.0"
    },
    {
      "url": "http://localhost:8080/a2a/order_agent",
      "protocolBinding": "JSONRPC",
      "protocolVersion": "0.3"
    }
  ]
}
```

Check it from the running container:

```bash
docker compose exec agent python -c \
  "import urllib.request; print(urllib.request.urlopen('http://localhost:8080/a2a/order_agent/.well-known/agent-card.json').read().decode())"
```

Both `quotation_agent` and `order_agent` expose cards and JSON-RPC endpoints:

| App | Card | RPC |
|---|---|---|
| `quotation_agent` | `/a2a/quotation_agent/.well-known/agent-card.json` | `/a2a/quotation_agent` |
| `order_agent` | `/a2a/order_agent/.well-known/agent-card.json` | `/a2a/order_agent` |

## 4. API cost

| Interaction | Model calls |
|---|---:|
| Order query containing a number | 1 (order classifier) |
| Order clarification turn (`我要查訂單`) | 1 |
| Follow-up bare number (`ORD-…`) | 0 (pending intent + direct path) |
| A2A card discovery / JSON-RPC | 0 |
| Quotation query (existing behavior) | 1 |
| Deterministic mode (`ORDER_AGENT_FAKE_LLM=1`) | 0 |

Before this change, an order query sent to the quotation agent also cost 1 classifier call and then refused. The new path moves that one call to the order agent, so there is **no net increase**. The A2A transport itself never calls a model.

## 5. Verification

- Unit tests: **35 passed** (`cd agent && uv run pytest tests/unit -q`), including the handoff detector and the `報價單轉訂單` boundary.
- Integration tests: **5 passed** (`MCP_URL=http://localhost:8010/mcp uv run pytest tests/integration -q`).
- Browser E2E: real-model single-turn handoff (screenshot 1.1) and deterministic multi-turn handoff (screenshots 1.2).
- A2A card: `200`, advertised URL reachable from the agent container.
- Docker: `agent`, `mcp-server`, `backend`, `db` healthy.
- Model-state check: the container is back on `AGENT_MODEL=gemini-3.1-flash-lite` with `ORDER_AGENT_FAKE_LLM` empty.
- Development budget: well under 50 model calls; the 503 retries were the bulk of the failed attempts.

## 6. Files changed

Core:

- `agent/quotation_agent/app_utils/a2a.py` — `A2A_BASE_URL`, `call_remote_agent()`.
- `agent/quotation_agent/fast_api_app.py` — publishes `/a2a/order_agent` using the real order graph.
- `agent/quotation_agent/agent.py` — `guard` route, `order_handoff`, edge map, Markdown-safe relay.
- `agent/quotation_agent/routing.py` — `detect_order_request()`.
- `agent/tests/unit/test_routing.py` — handoff/regression assertions.

Configuration and docs:

- `agent/.env.example`, `docker-compose.yml` — `A2A_BASE_URL`; model default `gemini-3.1-flash-lite`.
- `agent/README.md`, `README.md`, `README.zh-TW.md` — A2A section and model update.
- `order-agent-report.md` — boundary section replaced with the A2A handoff.

## 7. Reproduce

```bash
# Build the local agent (the published image does not contain this change)
docker compose -f docker-compose.yml -f docker-compose.build.yml up -d --build agent

# Open the normal quotation UI
#   http://localhost:3000/agent
# Send:
#   我要查訂單，訂單編號：ORD-20260118-001
```

Zero-cost repeatable demo:

```bash
ORDER_AGENT_FAKE_LLM=1 docker compose -f docker-compose.yml -f docker-compose.build.yml up -d agent
# ... demo ...
# back to the real model:
docker compose -f docker-compose.yml -f docker-compose.build.yml up -d agent
```

The A2A routes are served by `quotation_agent.fast_api_app:app` (the Compose entrypoint), not by `adk web` / `agents-cli playground`. For local uvicorn, set `A2A_BASE_URL` to that port (for example `A2A_BASE_URL=http://localhost:8099`).

## 8. Known limitations

- **Provider availability:** `gemini-3.1-flash-lite` returned `503 UNAVAILABLE` during part of the session. The order agent's fallback text is intentional and the A2A hop still succeeds; the deterministic mode avoids the dependency for a demo.
- **Same-process A2A:** both apps run in one container and the handoff calls the same uvicorn process over real HTTP. It demonstrates the protocol and card discovery, but it is not a separate deployment. A dedicated order-agent service would be the next step if independent scaling is required.
- **In-memory A2A task/session store:** state is lost on restart.
- **Auto-generated card:** the card contains the workflow's internal nodes as skills (16 entries). It is valid but noisy; a curated card would be a polish item.
- **Footer:** the A2A footer is a demo aid; it is a single string in `order_handoff` and can be removed without touching the protocol path.
