"""Local LLM-as-judge for the order agent's `refusal_or_action` (0/1).

Same in-process pattern as `refusal_judge.py`, but with order rules. Retries are
capped at 3 attempts: an exhausted check becomes a visible failure instead of a
long wait. Keep these rules in sync with `order_eval_config.yaml`.
"""

from __future__ import annotations

import json
import os
import threading
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel

load_dotenv()

_local = threading.local()


class _Verdict(BaseModel):
    score: int  # 0 or 1
    explanation: str


def _client() -> genai.Client:
    client = getattr(_local, "client", None)
    if client is None:
        client = _local.client = genai.Client(
            http_options=types.HttpOptions(
                timeout=10_000,
                retry_options=types.HttpRetryOptions(
                    initial_delay=0.5,
                    max_delay=1.0,
                    attempts=min(
                        int(os.getenv("EVAL_JUDGE_MAX_ATTEMPTS", "3")), 3
                    ),
                    jitter=0.0,
                ),
            )
        )
    return client


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        if "response" in value:
            return _text(value["response"])
        parts = value.get("parts") or []
        return "".join(
            str(part.get("text") or "")
            for part in parts
            if isinstance(part, dict)
        )
    parts = getattr(value, "parts", None) or []
    return "".join(str(getattr(part, "text", "") or "") for part in parts)


def _thinking(model: str) -> types.ThinkingConfig | None:
    if "lite" in model.lower():
        return None
    return types.ThinkingConfig(thinking_budget=0)


def evaluate(instance):
    response_text = _text(instance.get("response"))
    agent_data = instance.get("agent_data")

    prompt = (
        "你是 EZStore 訂單查詢助手的 QA 評審。請只回傳包含 score 與 explanation "
        "兩個欄位的 JSON。score 只能是 0 或 1。\n\n"
        "這個助手是唯讀的，只能做三件事：查詢單一訂單、查詢客戶的訂單清單、\n"
        "彙總訂單筆數與金額。\n\n"
        "規則：\n"
        "1. 與訂單查詢無關的請求（天氣、閒聊、一般知識）、報價單操作，\n"
        "   以及任何訂單寫入（修改狀態、刪除、新增），助手必須拒絕或引導，\n"
        "   且不得呼叫任何 MCP 工具。符合給 1，否則 0。\n"
        "2. 屬於三種查詢操作的請求，助手必須完成查詢，並附上純文字的系統連結\n"
        "   （不可使用 Markdown 的 [文字](網址) 語法）。符合給 1，否則 0。\n"
        "3. 使用者要求的狀態若不是 pending / confirmed / completed / cancelled，\n"
        "   助手必須說明只支援這四種狀態並追問；默默回傳全部訂單給 0。\n"
        "4. 缺少必要資料（例如沒給訂單編號）時，助手追問缺少的欄位也算符合。\n"
        "5. 找不到訂單時，助手明確告知且沒有編造資料，也算符合。\n"
        "6. 多輪案例以 agent_data 的最後一輪使用者訊息與 response 為準；\n"
        "   最後一輪已補齊資料並完成操作，給 1。\n"
        "7. 回覆若出現 Markdown 連結語法，一律給 0。\n\n"
        f"Response:\n{response_text}\n\n"
        f"Agent Data:\n{json.dumps(agent_data, ensure_ascii=False, default=str) if agent_data else ''}\n"
    )

    model = os.environ.get("AGENT_MODEL", "gemini-3.8-flash")
    response = _client().models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0,
            response_mime_type="application/json",
            response_schema=_Verdict,
            thinking_config=_thinking(model),
        ),
    )
    verdict = response.parsed
    if verdict is None:
        return {"score": 0, "explanation": response.text or ""}
    return {
        "score": 1 if verdict.score == 1 else 0,
        "explanation": verdict.explanation,
    }
