"""Deterministic check: the final response contains a plain-text system link.

Rules, in order:
1. Markdown link syntax [text](url) -> 0 (the spec requires plain text links).
2. Off-topic refusal (eval_case_id starts with off_topic, or the reply contains
   「只能協助」) -> 1; no link is needed.
3. Clarification (reply starts with 「請提供」 or contains
   「請告訴我您要對報價單」) -> 1.
4. Not found / honest failure (contains 「找不到」, 「操作失敗」 or
   「必須大於」) -> 1.
5. Otherwise the reply must contain http(s)://.../quotations/<id> or
   /orders/<id> -> 1, else 0.
"""

import re

_LINK = re.compile(r"https?://\S+/(quotations|orders)/\d+")
_MARKDOWN_LINK = re.compile(r"\[[^\]]+\]\(https?://[^)]+\)")


def _response_text(instance) -> str:
    response = instance.get("response") or {}
    if isinstance(response, str):
        return response
    parts = response.get("parts") or []
    return "".join(p.get("text", "") for p in parts if isinstance(p, dict))


def evaluate(instance):
    text = _response_text(instance)
    case_id = str(instance.get("eval_case_id", ""))

    if _MARKDOWN_LINK.search(text):
        return {
            "score": 0,
            "explanation": "回覆使用了 Markdown 連結語法，spec 要求純文字連結",
        }
    if case_id.startswith("off_topic") or "只能協助" in text:
        return {"score": 1, "explanation": "離題拒答，不需要系統連結"}
    if text.startswith("請提供") or "請告訴我您要對報價單" in text:
        return {"score": 1, "explanation": "追問缺少資料，不需要系統連結"}
    if any(marker in text for marker in ("找不到", "操作失敗", "必須大於")):
        return {"score": 1, "explanation": "查無資料或誠實回報失敗，不需要明細連結"}
    if _LINK.search(text):
        return {"score": 1, "explanation": "回覆含純文字系統連結"}
    return {"score": 0, "explanation": "回覆缺少純文字系統連結"}
