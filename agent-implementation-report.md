# EZStore ADK Agent — Implementation Report

**Scope:** Step 7 of the quickstart — implement the ADK agent in `agent/` to the instructor's
spec, then prove with data (Step 9 eval lab) that it actually behaves that way.
**Repo state:** `feature/ezstore-agent-develop`, agent code in `agent/quotation_agent/`.
**Date:** 2026-10-28

---

## 1. What was built

A graph-based ADK agent (`quotation_agent/agent.py`) that talks to the EZStore backend only
through the MCP server (local dev `MCP_URL=http://localhost:8010/mcp`, Compose
`http://mcp-server:8000/mcp`). The graph has exactly one LLM node — `classify_intent`, which
picks the route; tool selection, argument passing, clarification, refusal and the final `respond`
text are deterministic Python. `respond` emits plain-text system links.

| Spec requirement | Implementation | Verified by |
|---|---|---|
| 6 quotation operations (get, list by customer, duplicate, update factor, update total, convert to order) | one graph node per operation, each calling one MCP tool | actions eval: 6/6 cases `5 / 1 / 1` |
| Refuse anything outside quotation management | `refuse` node; no tool call, no data change | guardrails eval: 2/2 off-topic cases pass |
| Ask when information is missing, then continue | `clarify` path + multi-turn continuation | `clarify_then_complete` passes with `agent_turn_count = 2` |
| Plain-text links, never Markdown | `respond` never wraps URLs; Markdown requested by the user is honoured for layout only | `link_present` code metric = 1.0 on all 10 cases across both datasets (6 actions + 4 guardrails) |
| Edits must not touch the source quotation | duplicate/modify create a copy and the MCP tool text states the original is unchanged | UI: original still `0.95 / 6175`, copies carry `複製自` |

### Example usage (web UI)

![Agent in the web UI: quotation lookup answered with a plain-text system link, off-topic question refused](.screenshots/agentusage_example.png)

At `http://localhost:3000/agent`: `我要查詢報價單，報價單編號：QT-20260330-001` returns the
quotation fields, items and a plain-text `/quotations/12` link; the follow-up `今天台北天氣如何？`
is refused in the same conversation with no tool call and no data change.

---

## 2. How it was evaluated

`agents-cli eval run` over two datasets, graded by four metrics — two code checks, two LLM
judges — so a hard rule never depends on a judge's mood:

| Metric | Layer | Guards |
|---|---|---|
| `custom_response_quality` | Gemini judge (1–5) | overall answer quality |
| `refusal_or_action` | Gemini judge (0/1) | refuse out-of-scope; complete in-scope ops |
| `link_present` | code (0/1) | plain-text `/quotations/<id>` / `/orders/<id>` link |
| `agent_turn_count` | code | multi-turn really ran two turns |

---

## 3. Results

| Run | Cases | `custom_response_quality` | `link_present` | `refusal_or_action` |
|---|---|---|---|---|
| Guardrails (baseline) | 3 | 5.00 | 1.00 | 1.00 |
| Guardrails (+ not-found case) | 4 | 5.00 | 1.00 | 1.00 |
| **Actions (all 6 operations)** | **6** | **5.00** | **1.00** | **1.00** |
| Guardrails (deliberate Markdown-link break) | 4 | 4.50 | 0.75 | 0.75 |
| Unit + integration tests (`pytest tests/unit tests/integration`) | 16 tests | — | — | 16 passed |

Zero `num_cases_error` in every run. HTML reports: `agent/artifacts/grade_results/**/results_*.html`.

![Actions dataset: 6/6 cases, all four metrics 1.0, judge reasons visible](.screenshots/actions_results.png)

---

## 4. Evidence

**Services, agent loop and guardrails work**

![docker compose ps](.screenshots/dockercomposeps.png)
![Guardrails baseline, 3 cases](.screenshots/firstguardrailresults.png)

**Multi-turn and not-found behaviour**

![New not-found case added to the dataset](.screenshots/addedtestcase.png)
![4-case run: not_found scores 5/1/1, turn count 2 on the multi-turn case](.screenshots/firstandsecondguardrailresults.png)

**Traceability: the eval says it acted, the system shows it acted**

![Quotations list: 3 new drafts from the actions run, original unchanged](.screenshots/quotations.png)
![Original QT-20260330-001 still 0.95 / 6175](.screenshots/quotation12.png)
![Copy carries the 複製自 note and the requested 1.5 factor](.screenshots/copy_from_original.png)
![New order created by the convert-to-order case](.screenshots/orders.png)

**The evaluation catches a real error (break test)**

![Deliberate Markdown-link break in respond()](.screenshots/gitdiff_showtempered.png)
![Break caught: on the one case that returns a link, link_present and refusal_or_action drop to 0 while quality stays 3–5](.screenshots/thirdguardrailresults_tempered.png)

Reverted with `git checkout -- agent/quotation_agent/agent.py`; the 6/6 actions run above was
produced after the revert, which is the confirmation that scores return.

---

## 5. Trust findings

1. **A correct answer was scored as a failure.** `not_found_quotation` ("QT-20990101-001") was
   answered honestly — the agent said it could not find the quotation and invented nothing — yet
   `link_present` demanded a `/quotations/<id>` link (impossible for a non-existent record) and
   `refusal_or_action` rule 2 demanded the operation be completed. Both the code metric **and** the
   LLM judge had the same blind spot. Fixed in `tests/eval/link_presence.py` (not-found counts as
   pass) and judge rule 4; the case now scores `5 / 1 / 1`. Lesson: the grading rules need
   evaluating too, and reasons must be read case by case, not just the average.
2. **The generic quality judge is not a safety net.** After the deliberate Markdown-link break,
   `custom_response_quality` only dropped to 3.0/5 — the judge noticed and named the violation, but
   still gave a mid score — while `link_present` and `refusal_or_action` went to 0. The run average
   (4.5) still looks passable; only the hand-written rule and the code metric treated it as a hard
   failure.
3. **Model deviation:** `AGENT_MODEL=gemini-3.1-flash-lite` instead of the lab's `gemini-3.8-flash`
   (the latter intermittently returns `503 UNAVAILABLE`); scores are model-dependent and should be
   re-baselined if the model changes. The old retry policy (8 attempts, backoff up to 60s) turned one
   503 into a **233s** wait for a single prompt. `_build_model()` now caps each request at 10s and
   allows 3 attempts (~31.5s worst case), so a failing model surfaces an error instead of hanging.

---

## 6. Gaps not yet covered

- Ambiguous customer names (similar prefixes) and Chinese numerals such as "十萬" in price input.
- Concurrent edits of the same quotation by two users.
- `操作失敗：…` wording reads like a raw system error; the reply is truthful but unfriendly.
- Stability was observed implicitly (repeated guardrail runs gave identical scores) but no formal
  two-run side-by-side comparison was recorded.

---

## 7. Reproduce

```bash
cd agent
uv run pytest tests/unit tests/integration -q

cd ../ && docker compose exec db psql -U ezstore -f /scripts/seed-test-data.sql && cd agent
agents-cli eval run --dataset tests/eval/datasets/quotation-actions.json \
  --config tests/eval/eval_config.yaml --output artifacts/grade_results/actions
python3 ../scripts/eval_summary.py "$(ls -t artifacts/grade_results/actions/*.json | head -1)" --explain
```
