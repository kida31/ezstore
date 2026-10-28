# Order Inquiry Agent — Implementation Report

**Scope**: `agent/order_agent/` plus three read-only tools added to `mcp_server/server.py`.
It shares the existing `agent` container and port 8080 with the quotation agent
(`/apps/order_agent`). The Chat UI is the *same* page for both agents — **only the URL query
differs**: `/agent` loads `quotation_agent`, `/agent?app=order_agent` loads this one, and the
value is passed as the app name to `/apps/{app}/...` and `/run_sse`.
No new service, no backend change, no write path.

**Architecture**: ADK Graph Workflow. The only LLM node is `classify_intent`; guards, routing,
arguments, clarification, refusal and error handling are deterministic Python. Tools:
`get_order`, `list_orders`, `order_summary`. Status accepts only
`pending / confirmed / completed / cancelled`; anything else is explained and clarified instead of
silently returning every order. Off-topic requests, quotation operations, write requests and the
"bare order number after a clarification" turn are designed not to reach the model; in one automated
pass, the 8-turn acceptance set used 4 calls.

## Use cases

![Order agent: detail, customer list, status summary](.screenshots/order-agent-use-cases.png)

One conversation covers detail lookup, a customer's orders and a status summary; counts and totals
are computed in code, and links are plain text.

## Boundary with the quotation agent

![Quotation agent refusing an order lookup](.screenshots/order-agent-negative-quotation.png)

The same sentence, "我要查詢訂單，訂單編號：ORD-20260118-001", sent to `/agent` (quotation agent)
is refused — the two apps' guards are complementary. The reverse also holds: the order agent refuses
quotation operations and any order write (e.g. changing the status to "已出貨") and points the user
to the quotation agent or to manual processing in the UI.

## Preliminary automated checks

The figures below are outputs from automated runs. They have not been independently
re-checked, and should be read as an indication that the agent is expected to behave as described,
not as a verified guarantee.

| Check | Result (as reported by the automated run) |
|---|---|
| Unit + integration tests (no Gemini calls) | 38 passed |
| Acceptance set (8 turns) | 8/8 in one pass; 4 model calls |
| Eval (6 cases) | one run reported quality 5.00, link 1.00, refusal 1.00, status 1.00, turns 1.33 (1, 1, 2); 0 errors |
| Docker | 5 services reported healthy at run time; no new service added |

Raw outputs: `agent/artifacts/grade_results/order-{actions,guardrails}/*.html`,
`agent/artifacts/traces/order-*.json`; detail-page screenshot in
`.screenshots/order-agent-linked-detail.png`.

## Two findings worth recording

1. **The eval initially graded the wrong agent.** `agents-cli` reads
   `agents-cli-manifest.yaml` (`agent_directory: quotation_agent`), so in-process inference ignored
   `--app-name`. The first run produced quality 5.0 but refusal 0.0; reading the per-case reasons
   showed the replies were quotation-agent refusals. Switched to a local server plus
   `--url` and `--app-name order_agent`, and documented the commands in `agent/README.md`.
   A plausible average does not prove the right thing was graded.
2. **Retries and budget.** At most 3 attempts per model call, then a fixed manual-intervention
   reply; auth/MCP/seed failures are never retried. Our call log recorded 34 of 50 calls, including one
   failure (an 8 s timeout rejected by the API; raised to 10 s) and 9 calls spent grading the wrong
   agent.

**Read-only by design**: the agent is given no MCP tool that mutates data, and the
refusal/clarification paths are written to call no tools at all. A spot check after one run matched
the seed data; this was not an exhaustive audit.
