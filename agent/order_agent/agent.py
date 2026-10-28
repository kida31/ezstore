"""EZStore order inquiry agent (read-only).

Second ADK application in this repository. Like the quotation agent, the graph
has exactly one LLM node: `classify_intent`. Tool selection, argument passing,
clarification, refusal and the final response are deterministic Python.

Nothing in this module writes to the backend: the only MCP tools it calls are
`get_order`, `list_orders` and `order_summary`.
"""

from __future__ import annotations

import json
import os
import re
from enum import Enum
from typing import Any

from google.adk.agents import LlmAgent
from google.adk.agents.context import Context
from google.adk.apps import App
from google.adk.events.event import Event
from google.adk.models import Gemini, LlmCapabilities, LlmResponse
from google.adk.models.base_llm import BaseLlm
from google.adk.workflow import Workflow
from google.genai import types
from pydantic import BaseModel, Field

from quotation_agent.mcp_client import call_mcp_tool

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover - dotenv ships with google-adk
    pass


_MODEL_NAME = os.getenv("AGENT_MODEL", "gemini-3.8-flash")

ORDER_STATUSES: tuple[str, ...] = ("pending", "confirmed", "completed", "cancelled")
STATUS_LABELS: dict[str, str] = {
    "pending": "待處理",
    "confirmed": "已確認",
    "completed": "已完成",
    "cancelled": "已取消",
}
_STATUS_ALIASES: dict[str, str] = {
    "待處理": "pending",
    "未處理": "pending",
    "已確認": "confirmed",
    "確認": "confirmed",
    "已完成": "completed",
    "完成": "completed",
    "已取消": "cancelled",
    "取消": "cancelled",
}
# Words that name a status the backend does not have. Refusing them is the
# difference between "I filtered" and "I quietly returned everything".
_UNSUPPORTED_STATUS_RE = re.compile(r"已出貨|待出貨|出貨中|運送中|已送達|已付款|退款")

ORDER_NUMBER_RE = re.compile(r"ORD-\d{8}-\d{3}", re.IGNORECASE)
_CUSTOMER_RE = re.compile(r"客戶(?:名稱)?\s*[:：]\s*([^\n，,。；;]+)")

# Deliberately narrow: a false positive would refuse a legitimate query.
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
        re.compile(r"(?:新增|建立|修改|更新|刪除)(?:一個|一筆)?\s*(?:客戶|產品|商品)"),
        "客戶或產品的新增修改",
    ),
)

_QUOTATION_DOMAIN_RE = re.compile(r"報價單|報價因子|報價|quotation|QT-\d{8}-\d{3}", re.IGNORECASE)
_WRITE_RE = re.compile(
    r"(?:修改|更改|改為|改成|更新|變更|設定為|設成|刪除|取消)\s*(?:訂單|狀態)"
    r"|(?:訂單|狀態)\s*(?:修改|更改|改為|改成|更新|刪除|取消)"
    r"|(?:把|將)\s*ORD-\d{8}-\d{3}\s*(?:改成|改為|修改|刪除|取消)"
    r"|(?:刪除|新增|建立)\s*(?:一筆|一個)?\s*(?:訂單)",
    re.IGNORECASE,
)

MAX_LLM_CALLS = int(os.getenv("ORDER_AGENT_MAX_LLM_CALLS", "50"))
MAX_ATTEMPTS = min(int(os.getenv("ORDER_AGENT_MAX_ATTEMPTS", "3")), 3)
LLM_TIMEOUT_MS = max(int(os.getenv("ORDER_AGENT_LLM_TIMEOUT_MS", "10000")), 10_000)
FAKE_LLM = os.getenv("ORDER_AGENT_FAKE_LLM", "").strip().lower() in {"1", "true", "yes", "on"}

MANUAL_LLM_TEXT = (
    "目前無法完成查詢：模型服務暫時無法使用（已達嘗試或呼叫預算上限）。"
    "資料未被修改。請稍後手動重試，或聯絡管理員。"
)
MANUAL_MCP_TEXT = (
    "目前無法完成查詢：訂單系統暫時無法連線。資料未被修改。"
    "請稍後手動重試，或聯絡管理員。"
)
REFUSAL_TEXT = (
    "抱歉，我只能協助訂單查詢，例如查詢某一筆訂單、查詢客戶的訂單清單，"
    "或彙總訂單金額與狀態筆數。天氣、閒聊與其他系統操作我無法協助。"
)
REDIRECT_TEXT = (
    "這是報價單相關操作，不屬於訂單 Agent 的範圍。"
    "請改用報價單 Agent（http://localhost:3000/agent），它可以查詢、複製、"
    "調整報價單並轉成訂單。"
)
READONLY_TEXT = (
    "本 Agent 為唯讀，只能查詢訂單。系統目前也不支援透過對話變更訂單狀態，"
    "因此我不會執行任何修改。請以系統介面人工處理。"
)
CLARIFY_ORDER_TEXT = "請提供訂單編號（格式 ORD-YYYYMMDD-NNN），我再為您查詢。"
CLARIFY_STATUS_TEXT = (
    "目前只支援 pending（待處理）、confirmed（已確認）、completed（已完成）、"
    "cancelled（已取消）四種狀態。請告訴我要查詢哪一種狀態。"
)
UNCLEAR_TEXT = (
    "請告訴我您要查詢什麼：某一筆訂單（請給訂單編號）、某位客戶的訂單清單，"
    "或訂單金額與狀態彙總。"
)


def _env_true(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _thinking_config(model_name: str) -> types.ThinkingConfig | None:
    if "lite" in model_name.lower():
        return None
    return types.ThinkingConfig(thinking_budget=0)


def _build_model() -> Gemini:
    use_vertex = _env_true("GOOGLE_GENAI_USE_VERTEXAI")
    if (
        not use_vertex
        and _env_true("GOOGLE_GENAI_USE_ENTERPRISE")
        and not os.getenv("GEMINI_API_KEY")
    ):
        use_vertex = True

    client_kwargs: dict[str, Any] | None = None
    if use_vertex:
        client_kwargs = {"vertexai": True}
        project = os.getenv("GOOGLE_CLOUD_PROJECT")
        location = os.getenv("GOOGLE_CLOUD_LOCATION", "global")
        if project:
            client_kwargs["project"] = project
        if location:
            client_kwargs["location"] = location

    return Gemini(
        model=_MODEL_NAME,
        client_kwargs={
            **(client_kwargs or {}),
            "http_options": types.HttpOptions(timeout=LLM_TIMEOUT_MS),
        },
        # Hard cap: 3 attempts total, then the graph switches to a manual
        # intervention reply. Retries are never triggered by the agent itself.
        retry_options=types.HttpRetryOptions(
            initial_delay=0.5,
            max_delay=1.0,
            attempts=MAX_ATTEMPTS,
            jitter=0.0,
        ),
    )


# ---------------------------------------------------------------------------
# Structured intent
# ---------------------------------------------------------------------------


class IntentType(str, Enum):
    GET_ORDER = "get_order"
    LIST_ORDERS = "list_orders"
    ORDER_SUMMARY = "order_summary"
    OFF_TOPIC = "off_topic"
    UNCLEAR = "unclear"


class Intent(BaseModel):
    intent: IntentType = Field(
        description=(
            "One order-read operation, off_topic for unrelated requests, or "
            "unclear when the request is order-related but ambiguous."
        )
    )
    order_number: str | None = Field(
        default=None, description="Order number in the form ORD-YYYYMMDD-NNN."
    )
    customer_name: str | None = Field(
        default=None, description="Customer name for list_orders or order_summary."
    )
    status: str | None = Field(
        default=None,
        description="Optional status filter: pending/confirmed/completed/cancelled.",
    )
    reason: str = Field(default="", description="Short reason for off_topic/unclear.")


CLASSIFIER_INSTRUCTION = """\
你是 EZStore 訂單查詢系統的意圖分類器。你的唯一工作是將使用者的最新訊息
分類成固定類別並抽取參數。不要回答使用者，不要呼叫任何工具。

最新使用者訊息：
{current_message}

上一輪待完成的操作（可能為空）：
{pending_summary}

三個合法操作：
- get_order：查詢某一筆訂單的詳細內容（需要訂單編號 ORD-YYYYMMDD-NNN）。
- list_orders：查詢某位客戶的訂單清單，或依狀態篩選訂單。
- order_summary：彙總訂單筆數、各狀態數量或金額合計。

其他類別：
- off_topic：與訂單查詢無關，例如天氣、閒聊、報價單操作、訂單修改刪除、
  客戶或產品的新增修改。
- unclear：看起來與訂單有關，但無法判斷是哪一個操作。

規則：
1. 訂單編號格式是 ORD-YYYYMMDD-NNN。
2. 客戶名稱只抽名字，不要包含「客戶名稱：」或後面的「的訂單」。
3. 狀態只抽 pending、confirmed、completed、cancelled；其他狀態不要猜測。
4. 若上一輪有 pending intent，而最新訊息只是在補資料（例如只給訂單編號），
   請沿用 pending intent。
5. 「查詢訂單」但沒有給編號時，仍分類成 get_order；缺少的欄位由程式追問。
6. 任何修改、刪除、新增訂單的要求都不是合法操作，分類成 off_topic。
"""


def _last_user_text(llm_request: Any) -> str:
    for content in reversed(getattr(llm_request, "contents", []) or []):
        if getattr(content, "role", "") != "user":
            continue
        parts = getattr(content, "parts", []) or []
        text = "".join(str(getattr(part, "text", "") or "") for part in parts)
        if text.strip():
            return text.strip()
    return ""


def rule_intent(message: str) -> Intent:
    """Rule-based classifier used by tests and `ORDER_AGENT_FAKE_LLM=1`.

    It is intentionally the same decision table the prompt describes, so the
    graph can be exercised end-to-end without a model call.
    """
    text = message.strip()
    order_number = extract_order_number(text)
    customer = extract_customer_name(text)
    status, _ = detect_status(text)

    if re.search(r"彙總|統計|幾筆|多少錢|總金額|合計|各狀態", text):
        return Intent(intent=IntentType.ORDER_SUMMARY, customer_name=customer)
    if re.search(r"清單|列表|所有訂單|全部訂單|客戶.*訂單|訂單.*客戶", text):
        return Intent(
            intent=IntentType.LIST_ORDERS,
            customer_name=customer,
            status=status,
        )
    if order_number or re.search(r"訂單", text):
        return Intent(
            intent=IntentType.GET_ORDER,
            order_number=order_number,
        )
    return Intent(intent=IntentType.UNCLEAR)


class _FakeClassifierModel(BaseLlm):
    """Zero-cost stand-in for Gemini used by tests and local UI checks."""

    model: str = "rule-based"

    @property
    def capabilities(self) -> LlmCapabilities:
        return LlmCapabilities(output_schema_and_tools=True)

    async def generate_content_async(self, llm_request: Any, stream: bool = False):
        intent = rule_intent(_last_user_text(llm_request))
        yield LlmResponse(
            content=types.Content(
                role="model",
                parts=[types.Part.from_text(text=intent.model_dump_json())],
            )
        )


# ---------------------------------------------------------------------------
# Deterministic helpers
# ---------------------------------------------------------------------------


def extract_order_number(text: str) -> str | None:
    match = ORDER_NUMBER_RE.search(text or "")
    return match.group(0).upper() if match else None


def extract_customer_name(text: str) -> str | None:
    match = _CUSTOMER_RE.search(text or "")
    if not match:
        # Also accept "史塔克工業的訂單 ..." without an explicit label.
        match = re.search(r"([^\s，,。；;：:]{2,20})的(?:訂單|報價單)", text or "")
    if not match:
        return None
    name = re.sub(r"^(?:查詢|查看|列出|顯示|給我|幫我)", "", match.group(1).strip())
    name = re.sub(r"(的)?(訂單|清單).*$", "", name).strip()
    return name or None


def detect_status(text: str) -> tuple[str | None, str | None]:
    """Return (supported code, unsupported raw word)."""
    lowered = (text or "").lower()
    for code in ORDER_STATUSES:
        if re.search(rf"\b{code}\b", lowered):
            return code, None
    for alias, code in _STATUS_ALIASES.items():
        if alias in (text or ""):
            return code, None
    match = _UNSUPPORTED_STATUS_RE.search(text or "")
    if match:
        return None, match.group(0)
    return None, None


def deterministic_off_topic(message: str) -> str | None:
    for pattern, reason in _OFF_TOPIC_PATTERNS:
        if pattern.search(message):
            return reason
    return None


def is_quotation_request(message: str) -> bool:
    # An order-number question may legitimately mention the source quotation.
    if ORDER_NUMBER_RE.search(message or ""):
        return False
    return _QUOTATION_DOMAIN_RE.search(message or "") is not None


def is_write_request(message: str) -> bool:
    return _WRITE_RE.search(message or "") is not None


def _content_text(content: Any) -> str:
    if content is None:
        return ""
    parts = content.get("parts") if isinstance(content, dict) else getattr(content, "parts", None)
    chunks: list[str] = []
    for part in parts or []:
        if isinstance(part, dict):
            chunks.append(str(part.get("text") or ""))
        else:
            chunks.append(str(getattr(part, "text", "") or ""))
    return "".join(chunks).strip()


def _user_text(ctx: Context, node_input: Any) -> str:
    text = _content_text(ctx.user_content)
    if text:
        return text
    if isinstance(node_input, str):
        return node_input.strip()
    return _content_text(node_input)


def _infer_pending_from_history(ctx: Context) -> str | None:
    """Recover a pending get_order intent from a clarification turn."""
    events = list(getattr(ctx.session, "events", []) or [])
    clarify_index: int | None = None
    for index in range(len(events) - 1, -1, -1):
        if _content_text(events[index].content).startswith("請提供訂單編號"):
            clarify_index = index
            break
    if clarify_index is None:
        return None
    for index in range(clarify_index - 1, -1, -1):
        if getattr(events[index], "author", "") == "user":
            return "get_order"
    return None


def _as_intent(value: Any) -> Intent:
    if isinstance(value, Intent):
        return value
    if isinstance(value, dict):
        return Intent.model_validate(value)
    return Intent.model_validate_json(str(value))


def _pending_intent(value: Any) -> Intent | None:
    if not value:
        return None
    try:
        return Intent.model_validate(value)
    except Exception:  # noqa: BLE001 - stale state is ignored
        return None


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------

_llm_calls = 0


def reset_call_budget() -> None:
    """Test helper: a fresh process starts with a fresh budget."""
    global _llm_calls
    _llm_calls = 0


def llm_calls() -> int:
    return _llm_calls


def load_context(ctx: Context, node_input: Any) -> Event:
    message = _user_text(ctx, node_input)
    pending = _pending_intent(ctx.state.get("pending_intent"))
    if pending is None:
        inferred = _infer_pending_from_history(ctx)
        if inferred:
            pending = Intent(intent=IntentType.GET_ORDER)
    payload = pending.model_dump(mode="json") if pending else None
    summary = json.dumps(payload, ensure_ascii=False) if payload else "（無）"
    return Event(
        output=message,
        state={
            "current_message": message,
            "pending_intent": payload,
            "pending_summary": summary,
        },
    )


def guard(ctx: Context, node_input: str) -> Event:
    message = node_input
    kind = ""
    reason = ""
    if is_write_request(message):
        kind = "readonly"
        reason = "訂單寫入"
    elif is_quotation_request(message):
        kind = "quotation"
        reason = "報價單操作"
    else:
        off_topic = deterministic_off_topic(message)
        if off_topic:
            kind = "off_topic"
            reason = off_topic
    return Event(
        output=message,
        state={"guard_kind": kind, "guard_reason": reason},
    )


def dispatch(ctx: Context, node_input: str) -> Event:
    """Choose between deterministic handling and the single LLM node."""
    global _llm_calls
    kind = str(ctx.state.get("guard_kind") or "")
    if kind:
        return Event(output=node_input, route=kind)

    # A bare order number after a clarification needs no classification: the
    # pending intent already says what to do with it. This also saves a model
    # call on the most common follow-up turn.
    pending = _pending_intent(ctx.state.get("pending_intent"))
    number_only = re.fullmatch(
        r"\s*(ORD-\d{8}-\d{3})[。.！!]?\s*", node_input, re.IGNORECASE
    )
    if pending and pending.intent == IntentType.GET_ORDER and number_only:
        return Event(
            output={"intent": IntentType.GET_ORDER.value, "order_number": number_only.group(1).upper()},
            route="direct",
        )

    if _llm_calls >= MAX_LLM_CALLS:
        return Event(output=MANUAL_LLM_TEXT, route="manual")

    _llm_calls += 1
    return Event(output=node_input, route="llm")


def route_intent(ctx: Context, node_input: Any) -> Event:
    message = str(ctx.state.get("current_message") or "")
    try:
        classified = _as_intent(node_input)
    except Exception:  # noqa: BLE001 - malformed model output must not crash
        classified = Intent(intent=IntentType.UNCLEAR, reason="classifier output unparsable")

    pending = _pending_intent(ctx.state.get("pending_intent"))
    if classified.reason == "model_error":
        return Event(output=MANUAL_LLM_TEXT, route="manual", state={"pending_intent": None})

    if classified.intent in (IntentType.OFF_TOPIC, IntentType.UNCLEAR) and pending:
        classified = pending

    if classified.intent == IntentType.OFF_TOPIC:
        return Event(output=REFUSAL_TEXT, route="refuse", state={"pending_intent": None})

    # Deterministic extraction wins over model extraction: the model is not
    # allowed to invent a number or drop a status the user actually typed.
    order_number = extract_order_number(message) or classified.order_number
    customer_name = extract_customer_name(message) or classified.customer_name
    status_code, unsupported = detect_status(message)
    status_code = status_code or classified.status

    if classified.intent == IntentType.GET_ORDER:
        if not order_number:
            pending_payload = Intent(intent=IntentType.GET_ORDER).model_dump(mode="json")
            return Event(
                output={"text": CLARIFY_ORDER_TEXT, "pending": pending_payload},
                route="clarify",
                state={"pending_intent": pending_payload},
            )
        return Event(
            output={"order_number": order_number},
            route=IntentType.GET_ORDER.value,
            state={"pending_intent": None},
        )

    if unsupported:
        return Event(
            output={"text": CLARIFY_STATUS_TEXT, "pending": None},
            route="clarify",
            state={"pending_intent": None},
        )
    if status_code and status_code not in ORDER_STATUSES:
        return Event(
            output={"text": CLARIFY_STATUS_TEXT, "pending": None},
            route="clarify",
            state={"pending_intent": None},
        )

    if classified.intent == IntentType.LIST_ORDERS:
        return Event(
            output={
                "customer_name": customer_name or "",
                "status": status_code or "",
            },
            route=IntentType.LIST_ORDERS.value,
            state={"pending_intent": None},
        )

    if classified.intent == IntentType.ORDER_SUMMARY:
        return Event(
            output={"customer_name": customer_name or ""},
            route=IntentType.ORDER_SUMMARY.value,
            state={"pending_intent": None},
        )

    return Event(output={"text": UNCLEAR_TEXT, "pending": None}, route="clarify")


async def _run_tool(ctx: Context, tool_name: str, arguments: dict[str, Any]) -> Event:
    result = await call_mcp_tool(ctx, tool_name, arguments)
    text = result.text
    if result.is_error and "找不到" not in text:
        # Connectivity/tool failures go straight to the manual-intervention
        # message instead of being retried or paraphrased by the model.
        text = MANUAL_MCP_TEXT
    return Event(output=text, state={"pending_intent": None, "last_tool_error": result.is_error})


async def get_order(ctx: Context, node_input: dict[str, Any]) -> Event:
    return await _run_tool(ctx, "get_order", {"order_number": node_input["order_number"]})


async def list_orders(ctx: Context, node_input: dict[str, Any]) -> Event:
    return await _run_tool(
        ctx,
        "list_orders",
        {
            "customer_name": node_input.get("customer_name") or "",
            "status": node_input.get("status") or "",
        },
    )


async def order_summary(ctx: Context, node_input: dict[str, Any]) -> Event:
    return await _run_tool(
        ctx,
        "order_summary",
        {"customer_name": node_input.get("customer_name") or ""},
    )


async def clarify(ctx: Context, node_input: dict[str, Any]) -> Event:
    return Event(
        output=str(node_input.get("text") or UNCLEAR_TEXT),
        state={"pending_intent": node_input.get("pending")},
    )


async def refuse(ctx: Context, node_input: Any) -> Event:
    return Event(output=REFUSAL_TEXT, state={"pending_intent": None})


async def redirect(ctx: Context, node_input: Any) -> Event:
    return Event(output=REDIRECT_TEXT, state={"pending_intent": None})


async def readonly(ctx: Context, node_input: Any) -> Event:
    return Event(output=READONLY_TEXT, state={"pending_intent": None})


async def manual(ctx: Context, node_input: Any) -> Event:
    text = str(node_input or "") if isinstance(node_input, str) else ""
    return Event(output=text or MANUAL_LLM_TEXT, state={"pending_intent": None})


async def respond(ctx: Context, node_input: Any) -> Event:
    """Emit the only user-facing text event, under the fixed node name `respond`."""
    if isinstance(node_input, str):
        text = node_input.strip()
    elif node_input is None:
        text = ""
    else:
        text = json.dumps(node_input, ensure_ascii=False, default=str)
    if not text:
        text = "查詢完成，但系統沒有回傳內容。"
    content = types.Content(role="model", parts=[types.Part.from_text(text=text)])
    return Event(output=text, content=content, state={"last_response": text})


async def _on_model_error(callback_context: Any, llm_request: Any, error: Exception) -> LlmResponse:
    """Turn an exhausted model call into a routable sentinel, not a crash.

    ADK retries internally (3 attempts total). When those are gone, the graph
    must answer with the manual-intervention text instead of propagating a 429
    to the chat UI.
    """
    sentinel = Intent(intent=IntentType.UNCLEAR, reason="model_error")
    return LlmResponse(
        content=types.Content(
            role="model",
            parts=[types.Part.from_text(text=sentinel.model_dump_json())],
        )
    )


def build_app(fake_llm: bool | None = None) -> App:
    """Build the graph.

    ``fake_llm=True`` swaps Gemini for the rule-based model so the whole graph
    can be exercised with zero API calls (tests and `ORDER_AGENT_FAKE_LLM=1`).
    """
    use_fake = FAKE_LLM if fake_llm is None else fake_llm
    classifier = LlmAgent(
        name="classify_intent",
        model=_FakeClassifierModel() if use_fake else _build_model(),
        instruction=CLASSIFIER_INSTRUCTION,
        output_schema=Intent,
        on_model_error_callback=_on_model_error,
        generate_content_config=types.GenerateContentConfig(
            temperature=0.0,
            thinking_config=_thinking_config(_MODEL_NAME),
        ),
    )

    edges = [
        ("START", load_context),
        (load_context, guard),
        (guard, dispatch),
        (
            dispatch,
            {
                "llm": classifier,
                "direct": route_intent,
                "off_topic": refuse,
                "quotation": redirect,
                "readonly": readonly,
                "manual": manual,
            },
        ),
        (classifier, route_intent),
        (
            route_intent,
            {
                IntentType.GET_ORDER.value: get_order,
                IntentType.LIST_ORDERS.value: list_orders,
                IntentType.ORDER_SUMMARY.value: order_summary,
                "clarify": clarify,
                "refuse": refuse,
                "manual": manual,
            },
        ),
        (get_order, respond),
        (list_orders, respond),
        (order_summary, respond),
        (clarify, respond),
        (refuse, respond),
        (redirect, respond),
        (readonly, respond),
        (manual, respond),
    ]

    root_agent = Workflow(
        name="order_agent",
        description=(
            "EZStore order inquiry agent (read-only): look up an order by number, "
            "list a customer's orders, or summarise order counts and totals."
        ),
        edges=edges,
    )
    return App(name="order_agent", root_agent=root_agent)


app = build_app()
root_agent = app.root_agent
