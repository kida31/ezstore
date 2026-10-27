"""Deterministic routing helpers.

The LLM only classifies intent and extracts fields. Every decision about which
tool to call, whether to clarify, and whether to refuse is made here in code.
"""

from __future__ import annotations

import re

from quotation_agent.models import Intent, IntentType, OPERATION_INTENTS

QUOTATION_NUMBER_RE = re.compile(r"QT-\d{8}-\d{3}", re.IGNORECASE)
_NUMBER_RE = re.compile(r"^[0-9]+(?:\.[0-9]+)?$")
_FACTOR_RE = re.compile(
    r"(?:報價因子|pricing\s*factor|factor)\s*"
    r"(?:改為|調整為|調到|調整到|設為|變成|到|[:：=])?\s*"
    r"([0-9]+(?:\.[0-9]+)?)",
    re.IGNORECASE,
)
_TOTAL_RE = re.compile(
    r"(?:目標總價|總價|total(?:\s*price)?)\s*"
    r"(?:改為|調整為|調到|調整到|設為|變成|到|[:：=])?\s*"
    r"([0-9]+(?:\.[0-9]+)?)",
    re.IGNORECASE,
)
_CUSTOMER_RE = re.compile(r"客戶(?:名稱)?\s*[:：]\s*([^\n，,。；;]+)")

# These are deliberately narrow. A false positive would refuse a legitimate
# quotation request, so only unmistakable out-of-scope patterns are listed.
_OFF_TOPIC_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"天氣|氣溫|下雨|晴天|weather", re.IGNORECASE), "天氣"),
    (
        re.compile(
            r"^\s*(?:你好|哈囉|嗨|hello|hi|早安|午安|晚安)[！!。.，,~\s]*$",
            re.IGNORECASE,
        ),
        "閒聊",
    ),
    (
        re.compile(
            r"笑話|閒聊|聊天|陪我聊|你是誰|你今天|心情|電影|音樂|星座|旅遊|故事",
            re.IGNORECASE,
        ),
        "閒聊",
    ),
    (
        re.compile(
            r"(?:新增|建立|修改|更新|刪除|移除|編輯).{0,8}(?:客戶|產品)",
            re.IGNORECASE,
        ),
        "客戶或產品的新增修改",
    ),
    (
        re.compile(
            r"(?:客戶|產品).{0,8}(?:新增|建立|修改|更新|刪除|移除|編輯)",
            re.IGNORECASE,
        ),
        "客戶或產品的新增修改",
    ),
    (
        re.compile(
            r"(?:新增|建立|修改|更新|刪除|移除|編輯).{0,8}訂單",
            re.IGNORECASE,
        ),
        "訂單的新增修改",
    ),
    (
        re.compile(
            r"(?:訂單).{0,8}(?:新增|建立|修改|更新|刪除|移除|編輯)",
            re.IGNORECASE,
        ),
        "訂單的新增修改",
    ),
    (
        re.compile(r"(?:刪除|移除).{0,6}報價單", re.IGNORECASE),
        "報價單刪除",
    ),
    (
        re.compile(
            r"忽略(?:之前|以上|所有)|ignore\s+(?:previous|all)|"
            r"系統提示|system\s+prompt|角色扮演|扮演|pretend",
            re.IGNORECASE,
        ),
        "提示詞注入",
    ),
)


def normalize_quotation_number(value: str | None) -> str | None:
    if not value:
        return None
    match = QUOTATION_NUMBER_RE.search(value)
    if match:
        return match.group(0).upper()
    cleaned = value.strip().strip("，,。.；;：: ")
    return cleaned.upper() or None


def extract_quotation_number(text: str) -> str | None:
    match = QUOTATION_NUMBER_RE.search(text or "")
    return match.group(0).upper() if match else None


def extract_pricing_factor(text: str) -> float | None:
    match = _FACTOR_RE.search(text or "")
    if match:
        return float(match.group(1))
    stripped = (text or "").strip()
    if _NUMBER_RE.fullmatch(stripped):
        return float(stripped)
    return None


def extract_target_total(text: str) -> float | None:
    match = _TOTAL_RE.search(text or "")
    if match:
        return float(match.group(1))
    stripped = (text or "").strip()
    if _NUMBER_RE.fullmatch(stripped):
        return float(stripped)
    return None


def _clean_customer_name(value: str) -> str | None:
    cleaned = re.split(r"(?:的)?報價單|的|，|,|。|；|;", value, maxsplit=1)[0]
    cleaned = cleaned.strip(" ：:，,。.；;")
    return cleaned or None


def extract_customer_name(text: str) -> str | None:
    match = _CUSTOMER_RE.search(text or "")
    if not match:
        return None
    return _clean_customer_name(match.group(1))


def deterministic_off_topic(text: str) -> str | None:
    for pattern, reason in _OFF_TOPIC_PATTERNS:
        if pattern.search(text or ""):
            return reason
    return None


def infer_pending_from_user_text(text: str) -> Intent | None:
    """Infer the operation from the user turn before a clarification.

    Eval traces seed conversation events, not the session state, so this lets a
    multi-turn case continue correctly even when `pending_intent` was never
    persisted in the new session.
    """
    raw = text or ""
    intent: IntentType | None = None
    if "轉訂單" in raw or "轉為訂單" in raw or "轉成訂單" in raw:
        intent = IntentType.CONVERT_TO_ORDER
    elif "複製" in raw:
        intent = IntentType.DUPLICATE_QUOTATION
    elif "報價因子" in raw:
        intent = IntentType.UPDATE_PRICING_FACTOR
    elif "總價" in raw:
        intent = IntentType.ADJUST_TOTAL_PRICE
    elif "客戶" in raw and ("清單" in raw or "報價單" in raw):
        intent = IntentType.LIST_CUSTOMER_QUOTATIONS
    elif "查詢" in raw and "報價單" in raw:
        intent = IntentType.GET_QUOTATION

    if intent is None:
        return None

    return Intent(
        intent=intent,
        quotation_number=extract_quotation_number(raw),
        customer_name=extract_customer_name(raw),
        new_pricing_factor=(
            extract_pricing_factor(raw)
            if intent == IntentType.UPDATE_PRICING_FACTOR
            else None
        ),
        target_total_price=(
            extract_target_total(raw)
            if intent == IntentType.ADJUST_TOTAL_PRICE
            else None
        ),
        reason="inferred from conversation history",
    )


def looks_like_continuation(message: str, pending: Intent) -> bool:
    text = message or ""
    if extract_quotation_number(text):
        return True
    if pending.intent == IntentType.UPDATE_PRICING_FACTOR:
        return _FACTOR_RE.search(text) is not None or _NUMBER_RE.fullmatch(
            text.strip()
        ) is not None
    if pending.intent == IntentType.ADJUST_TOTAL_PRICE:
        return _TOTAL_RE.search(text) is not None or _NUMBER_RE.fullmatch(
            text.strip()
        ) is not None
    if pending.intent == IntentType.LIST_CUSTOMER_QUOTATIONS:
        if extract_customer_name(text):
            return True
        stripped = text.strip(" ，,。.!！?？")
        return (
            1 <= len(stripped) <= 40
            and not any(
                keyword in stripped
                for keyword in (
                    "查詢",
                    "清單",
                    "複製",
                    "修改",
                    "轉",
                    "報價單",
                    "訂單",
                    "客戶",
                    "新增",
                    "刪除",
                )
            )
        )
    return False


def merge_intent(
    classified: Intent,
    pending: Intent | None,
    message: str,
) -> Intent:
    """Combine the classifier result with pending state and regex fallbacks."""
    result = classified.model_copy(deep=True)

    if (
        pending is not None
        and result.intent in (IntentType.OFF_TOPIC, IntentType.UNCLEAR)
        and looks_like_continuation(message, pending)
    ):
        result.intent = pending.intent

    if pending is not None and result.intent == pending.intent:
        for field in (
            "quotation_number",
            "customer_name",
            "new_pricing_factor",
            "target_total_price",
        ):
            if getattr(result, field) is None:
                setattr(result, field, getattr(pending, field))

    if result.intent not in OPERATION_INTENTS:
        return result

    quotation_number = extract_quotation_number(message)
    if quotation_number:
        result.quotation_number = quotation_number
    else:
        result.quotation_number = normalize_quotation_number(
            result.quotation_number
        )

    if result.intent == IntentType.LIST_CUSTOMER_QUOTATIONS:
        customer_name = extract_customer_name(message)
        if customer_name is None and pending is not None and (
            pending.intent == IntentType.LIST_CUSTOMER_QUOTATIONS
            and looks_like_continuation(message, pending)
        ):
            customer_name = _clean_customer_name(message)
        if customer_name:
            result.customer_name = customer_name

    if result.intent == IntentType.UPDATE_PRICING_FACTOR:
        factor = extract_pricing_factor(message)
        if factor is not None:
            result.new_pricing_factor = factor

    if result.intent == IntentType.ADJUST_TOTAL_PRICE:
        target = extract_target_total(message)
        if target is not None:
            result.target_total_price = target

    return result


def missing_fields(intent: Intent) -> list[str]:
    if intent.intent in (
        IntentType.GET_QUOTATION,
        IntentType.DUPLICATE_QUOTATION,
        IntentType.CONVERT_TO_ORDER,
    ):
        return [] if intent.quotation_number else ["quotation_number"]

    if intent.intent == IntentType.LIST_CUSTOMER_QUOTATIONS:
        return [] if intent.customer_name else ["customer_name"]

    if intent.intent == IntentType.UPDATE_PRICING_FACTOR:
        missing: list[str] = []
        if not intent.quotation_number:
            missing.append("quotation_number")
        if intent.new_pricing_factor is None:
            missing.append("new_pricing_factor")
        return missing

    if intent.intent == IntentType.ADJUST_TOTAL_PRICE:
        missing = []
        if not intent.quotation_number:
            missing.append("quotation_number")
        if intent.target_total_price is None:
            missing.append("target_total_price")
        return missing

    return []


_FIELD_LABELS = {
    "quotation_number": "報價單編號（格式：QT-YYYYMMDD-NNN）",
    "customer_name": "客戶名稱",
    "new_pricing_factor": "新的報價因子（大於 0 的數字）",
    "target_total_price": "目標總價（大於 0 的數字）",
}


def clarify_message(intent_value: str | None, missing: list[str]) -> str:
    if not intent_value or intent_value == IntentType.UNCLEAR.value:
        return (
            "請告訴我您要對報價單做什麼操作：查詢、查詢客戶清單、複製、"
            "修改報價因子、修改總價，或轉為訂單。"
        )

    labels = [_FIELD_LABELS[field] for field in missing if field in _FIELD_LABELS]
    if not labels:
        return "請提供要操作的報價單相關資料。"
    if len(labels) == 1:
        return f"請提供{labels[0]}。"
    return "請提供以下資料：" + "、".join(labels) + "。"


def is_clarification_text(text: str) -> bool:
    return "請提供" in text or "請告訴我您要對報價單" in text
