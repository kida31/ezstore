# EZStore Quotation Agent

Natural-language quotation management agent for EZStore, built with the Google
Agent Development Kit (ADK) 2.x graph `Workflow` API. It connects to the EZStore
MCP Server over Streamable HTTP and exposes six quotation operations plus
deterministic clarification and refusal paths.

## Design

The graph has **exactly one LLM node**: `classify_intent`. The classifier only
returns structured `Intent` JSON. Tool selection, tool arguments, missing-field
checks, clarification and refusal are all decided by Python code.

```
START
  → load_context          read the user message and pending operation
  → guard                 obvious off-topic requests (deterministic)
  → classify_intent       the only LLM call; Pydantic output_schema=Intent
  → route_intent          merge with pending state, validate required fields
      ├─ get_quotation
      ├─ list_customer_quotations
      ├─ duplicate_quotation
      ├─ update_pricing_factor
      ├─ adjust_total_price
      ├─ convert_to_order
      ├─ clarify           ask for missing fields; keep pending_intent
      └─ refuse            fixed refusal text; no tool call
  → respond               emit the final text and update chat_history
```

Every user message starts a new graph run. Cross-turn memory is kept in ADK
session state:

- `chat_history` — recent user/assistant turns (for context and eval traces)
- `pending_intent` — the operation waiting for missing fields
- `last_tool_error` / `last_response` — diagnostics and the emitted reply

The application name is `quotation_agent`, registered with
`App(name="quotation_agent", root_agent=...)` so the Chat UI, A2A and
`agents-cli` all address the same app.

## Project layout

```
agent/
├── quotation_agent/
│   ├── agent.py          Workflow nodes, edges and the App
│   ├── models.py         Intent enum and Pydantic schema
│   ├── prompts.py        classifier instruction and fixed refusal
│   ├── routing.py        deterministic merge, extraction and clarification
│   ├── mcp_client.py     McpToolset + StreamableHTTPConnectionParams
│   └── fast_api_app.py   FastAPI entrypoint, /health, in-memory services
├── tests/
│   ├── unit/             routing/state tests (no LLM, no MCP)
│   ├── integration/      graph and MCP integration tests
│   └── eval/             datasets, eval_config.yaml and local LLM judges
├── Dockerfile            production container, port 8080
└── pyproject.toml        Python 3.12, google-adk[mcp,a2a]
```

## Environment variables

Copy `agent/.env.example` to `agent/.env` for local development. The root
`.env` is used by Docker Compose.

| Variable | Purpose |
|----------|---------|
| `GEMINI_API_KEY` | Google AI Studio API key. Use either this or the Vertex AI settings below. |
| `AGENT_MODEL` | Gemini model name. Default `gemini-3.8-flash`; if 3.8 returns `503 high demand`, set `gemini-3.5-flash`. |
| `MCP_URL` | MCP endpoint. Local: `http://localhost:8010/mcp`; Compose: `http://mcp-server:8000/mcp`. |
| `FRONTEND_URL` | Public frontend base used in plain-text links (`http://localhost:3000` locally). |
| `ALLOW_ORIGINS` | CORS origins for the agent API. |
| `GOOGLE_GENAI_USE_VERTEXAI` | Set `true` to use Vertex AI instead of AI Studio. |
| `GOOGLE_CLOUD_PROJECT` | Vertex AI project ID. |
| `GOOGLE_CLOUD_LOCATION` | Vertex AI location; `global` in the example. |

Without credentials the agent still starts. `GET /health` stays healthy and
only chat requests fail, with a clear “missing API key” error; Compose does not
fail because of the agent.

## Local development

Start the database, backend and MCP Server first (the MCP port is 8010 so it
does not collide with the agent's local 8080):

```bash
docker compose up -d db backend mcp-server
docker compose exec -T db psql -U ezstore -f /scripts/seed-test-data.sql
```

Then run the agent:

```bash
cd agent
cp .env.example .env                 # fill GEMINI_API_KEY; set MCP_URL
uv sync --group dev --extra eval     # dev tools + eval metrics

agents-cli playground --port 8090    # web UI: http://127.0.0.1:8090
agents-cli run "我要查詢報價單，報價單編號：QT-20260330-001"
```

`agents-cli playground` defaults to port 8080, which the backend already uses
in this project. Use `--port 8090` locally.

### Tests

```bash
cd agent
uv run pytest tests/unit -q                    # routing/state, no LLM/MCP
MCP_URL=http://localhost:8010/mcp \
  uv run pytest tests/integration -q           # graph + MCP; LLM tests need a key
uv run pytest -q                               # everything configured locally
```

### Evaluation

The eval datasets and metric configuration live in `tests/eval/`:

| Metric | Kind | What it checks |
|--------|------|----------------|
| `custom_response_quality` | LLM judge, 1–5 | Accuracy, relevance and clarity of the final reply |
| `link_present` | Programmatic, 0/1 | Plain-text `/quotations/<id>` or `/orders/<id>` link; Markdown links fail; `off_topic*` cases are exempt |
| `refusal_or_action` | LLM judge, 0/1 | Off-topic requests are refused without tools; in-scope requests complete and use plain-text links; clarification/not-found responses count |
| `agent_turn_count` | Programmatic | Number of agent turns in the trace |

Run one dataset directly:

```bash
cd agent
agents-cli eval run \
  --dataset tests/eval/datasets/quotation-actions.json \
  --config tests/eval/eval_config.yaml
```

The action dataset creates new quotations and an order; reload
`/scripts/seed-test-data.sql` before rerunning it. The guardrails dataset
contains the two refusal cases and the multi-turn clarification case.

`refusal_or_action` is implemented by a local LLM judge
(`tests/eval/refusal_judge.py`) so an AI Studio key is enough for the judge. Its
prompt uses `{response}` and `{agent_data}`, never `{prompt}`, because
multi-turn cases have no top-level `prompt` field.

The full self-assessment suite is run from the repository root:

```bash
python3 scripts/grading/grade_agent.py --suite scripts/grading/student-suite.json
```

> **Eval CLI note:** `agents-cli` 1.11 constructs a Gemini platform client for
> the eval orchestrator even when every metric runs in-process. If `eval` asks
> for a GCP project, authenticate ADC (`gcloud auth application-default login`)
> and set `GOOGLE_CLOUD_PROJECT`; the agent model and the local judges can still
> use `GEMINI_API_KEY`. A CLI version whose `eval grade` supports local-only
> metrics avoids the extra GCP requirement.

## Docker

The Compose `agent` service is internal-only (no host port) and depends on
`mcp-server`; the frontend's nginx proxies `/agent-api/` to `agent:8080`.
Build and start everything with:

```bash
docker compose -f docker-compose.yml -f docker-compose.build.yml up -d --build
```

## Adding an operation

The LLM must not choose tools. Add the operation in these places:

1. **Schema** — add an enum member to `quotation_agent/models.py` and any
   operation-specific fields to `Intent`.
2. **Extraction/routing** — add deterministic field extraction, missing-field
   rules, clarification text and safety-net patterns in
   `quotation_agent/routing.py`.
3. **Classifier contract** — list the new intent in the allowed intents in
   `quotation_agent/prompts.py`. The classifier still only returns JSON.
4. **Graph node** — add an async node in `quotation_agent/agent.py` that calls
   one MCP tool, register it in the `route_intent` edge map, and connect it to
   `respond`.
5. **MCP contract** — add the exact tool name/argument shape to
   `quotation_agent/mcp_client.py` tests; the MCP Server owns the tool.
6. **Tests** — add unit tests for routing/clarification and an integration test
   for the MCP call.
7. **Eval** — add a case to `tests/eval/datasets/quotation-actions.json` (and
   any suite checks in `scripts/grading/student-suite.json`).

Keep these invariants: one LLM node, code-decided tool arguments, copy-on-write
for every modification, plain-text links only, and a clarification turn instead
of guessing missing fields.
