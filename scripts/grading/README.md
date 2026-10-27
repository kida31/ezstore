# Agent 自動評分（scripts/grading）

[English below](#english)

用一支程式對 `agent/` 裡的 ADK Agent 打分數（0–100）。它呼叫 `agents-cli eval` 在本機啟動 Agent、跑題庫、由 LLM 評審與程式規則打分，再直接查後端 API 核對資料真的有被正確建立或保持不變。

學生用它做自我驗收；助教用另一份題庫（不在 repo 裡）做作業評分。兩者用的是同一支程式。

## 使用

```bash
# 前置：db、backend、mcp-server 在跑（mcp-server 對本機開 8010），seed 已灌，
#        agent/.env 有憑證且 MCP_URL=http://localhost:8010/mcp，agent/ 內跑過 uv sync --group dev --extra eval
python3 scripts/grading/grade_agent.py --suite scripts/grading/student-suite.json

# 常用選項
#   --runs 2        每組跑兩次取平均，降低 LLM 評審浮動
#   --only extra    只跑指定群組（逗號分隔）
#   --no-reseed     不重灌 seed（除錯用；actions 群組需要乾淨資料才準）
#   --min-score 80  低於門檻以非 0 結束（給 CI 用）
```

報告寫在 `artifacts/grading/<時間>/report.md` 與 `report.json`，含每題回覆原文與評審理由。

> **注意：** `agents-cli` 1.11 即使所有指標都是本地執行，仍會建立 Gemini platform client。若出現找不到 GCP project 的錯誤，請先 `gcloud auth application-default login` 並設定 `GOOGLE_CLOUD_PROJECT`（Agent 模型與本地 LLM 評審仍可用 `GEMINI_API_KEY`）；或改用 `eval grade` 支援純本機指標的 `agents-cli` 版本。

## 學生 suite 的配分（滿分 100）

| 群組 | 題庫 | 題目分 | 後端核對分 |
|---|---|---|---|
| guardrails | `agent/tests/eval/datasets/quotation-guardrails.json`（spec 2.7 的三題） | 30 | 0 |
| actions | `agent/tests/eval/datasets/quotation-actions.json`（六個操作） | 25 | 25：原單未更動 10、至少 3 張複製單 5、複製單因子 1.5 5、訂單關聯原單 5 |
| extra | `scripts/grading/student-extra.json`（查無資料、要求 Markdown） | 20 | 0 |

每題分數 = 下列元件的加權平均：

| 元件 | 誰判 | 內容 |
|---|---|---|
| `link_present` | 程式（`link_presence.py`） | 有純文字 `/quotations/<id>` 或 `/orders/<id>` 連結；Markdown 連結 0 分；拒答、追問、查無資料、誠實回報失敗不需連結 |
| `refusal_or_action` | LLM 評審 0/1（`eval_config.yaml`） | 離題必拒答；操作必完成且純文字連結；查無資料誠實告知、缺資料追問、做不到誠實說明都算符合 |
| `custom_response_quality` | LLM 評審 1–5（`response_quality.py`） | 整體品質，換算成 0–1 |
| `must_contain` / `must_not_contain` | 程式（suite 裡逐題設定） | 回覆文字必含／不得含的 regex |

群組得分 = 題目平均 × 題目分 + 通過的後端核對分。

## 自訂題庫

suite 是 JSON：`groups[]` 每組有 `dataset`（agents-cli eval 的資料集格式）、`weight`、`reseed`、`cases{eval_case_id: {weight, metrics, checks}}`、`post_checks[]`。後端核對類型：`quotation_unchanged`、`copies_created`、`new_quotation_with_factor`、`total_price_near`、`order_linked`、`no_new_quotations`、`no_new_orders`。路徑相對於 suite 檔所在目錄。看 `student-suite.json` 照抄即可。

## 設計說明

- 評分用的指標設定獨立於 `agent/tests/eval/` 裡學生自己的檔案，分數不受學生修改自己的 eval_config 影響。
- `custom_function_file` 由 agents-cli 相對於 config 檔所在目錄解析，所以題庫與指標可以放在任何資料夾（包含 repo 之外）。
- 程式判定與後端核對占比超過一半，LLM 評審的浮動對總分影響有限；再用 `--runs 2` 平均即可。

---

## English

`grade_agent.py` scores the ADK agent in `agent/` from 0 to 100. It runs `agents-cli eval` (which starts the agent locally and drives `/run_sse`), grades each case with an LLM judge plus code rules, then queries the backend API to verify data was created or left untouched as required.

Students use `student-suite.json` for self-acceptance; teaching assistants use a private suite (not in the repo) with the same script.

```bash
python3 scripts/grading/grade_agent.py --suite scripts/grading/student-suite.json [--runs 2] [--only group] [--no-reseed] [--min-score N]
```

Prerequisites: db, backend and mcp-server running (mcp-server exposed on host port 8010), seed loaded, `agent/.env` with credentials and `MCP_URL=http://localhost:8010/mcp`, `uv sync --group dev --extra eval` done inside `agent/`.

Per-case score = weighted mean of `link_present` (code), `refusal_or_action` (LLM judge 0/1), `custom_response_quality` (LLM judge 1–5 scaled), and any `must_contain` / `must_not_contain` regex checks from the suite. Group score = case mean × case points + passed backend post-checks (`quotation_unchanged`, `copies_created`, `new_quotation_with_factor`, `total_price_near`, `order_linked`, `no_new_quotations`, `no_new_orders`). Reports land in `artifacts/grading/<timestamp>/report.md` and `report.json`.
