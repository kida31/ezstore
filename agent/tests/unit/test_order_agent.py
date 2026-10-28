"""Zero-API tests for the order agent.

Everything here runs without a Gemini call: the pure functions are tested
directly, and the full graph is exercised with `build_app(fake_llm=True)` plus a
stubbed MCP call.
"""

from __future__ import annotations

import pytest
from google.adk.runners import InMemoryRunner

import order_agent.agent as a
from quotation_agent.mcp_client import ToolResult


@pytest.fixture(autouse=True)
def _fresh_budget():
    a.reset_call_budget()
    yield
    a.reset_call_budget()


# ---------------------------------------------------------------------------
# Deterministic helpers
# ---------------------------------------------------------------------------


def test_extracts_order_number() -> None:
    assert a.extract_order_number("訂單編號：ord-20260118-001。") == "ORD-20260118-001"


def test_extracts_customer_name_with_label() -> None:
    assert a.extract_customer_name("客戶名稱：史塔克工業 的訂單") == "史塔克工業"


def test_extracts_customer_name_from_possessive() -> None:
    assert a.extract_customer_name("史塔克工業的訂單各狀態有幾筆？") == "史塔克工業"


def test_status_codes_and_labels() -> None:
    assert a.detect_status("查詢 pending 的訂單") == ("pending", None)
    assert a.detect_status("查詢已完成的訂單") == ("completed", None)
    assert a.detect_status("查詢已出貨的訂單") == (None, "已出貨")


def test_guard_off_topic() -> None:
    assert a.deterministic_off_topic("今天台北天氣如何？") == "天氣"


def test_guard_quotation_domain() -> None:
    assert a.is_quotation_request("幫我複製報價單 QT-20260330-001") is True
    # An order question may legitimately mention its source quotation.
    assert a.is_quotation_request("ORD-20260330-001 來自哪張報價單？") is False


def test_guard_write_request() -> None:
    assert a.is_write_request("我要把 ORD-20260322-001 改成已出貨") is True
    assert a.is_write_request("刪除訂單 ORD-20260322-001") is True
    assert a.is_write_request("查詢 ORD-20260322-001") is False


# ---------------------------------------------------------------------------
# Routing decisions
# ---------------------------------------------------------------------------


def _route(message: str, intent: dict, pending: dict | None = None):
    return a.route_intent(
        _ctx(message, pending),
        intent,
    )


def _ctx(message: str, pending: dict | None = None):
    class _State(dict):
        pass

    class _Ctx:
        state = _State(
            {"current_message": message, "pending_intent": pending}
        )

    return _Ctx()


def test_missing_order_number_clarifies_and_keeps_pending() -> None:
    event = _route("我要查詢訂單", {"intent": "get_order"})
    assert event.actions.route == "clarify"
    assert event.output["text"].startswith("請提供訂單編號")
    assert event.actions.state_delta["pending_intent"]["intent"] == "get_order"


def test_pending_intent_is_resumed_by_order_number() -> None:
    pending = {"intent": "get_order"}
    event = _route("ORD-20260322-001", {"intent": "unclear"}, pending)
    assert event.actions.route == "get_order"
    assert event.output["order_number"] == "ORD-20260322-001"


def test_unsupported_status_is_rejected_not_ignored() -> None:
    event = _route(
        "查詢已出貨的訂單",
        {"intent": "list_orders", "customer_name": "史塔克工業"},
    )
    assert event.actions.route == "clarify"
    assert "pending（待處理）" in event.output["text"]


def test_list_orders_passes_status_code() -> None:
    event = _route("查詢已完成的訂單", {"intent": "list_orders"})
    assert event.actions.route == "list_orders"
    assert event.output["status"] == "completed"


# ---------------------------------------------------------------------------
# Failure handling: no retries, manual intervention
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tool_failure_becomes_manual_message(monkeypatch) -> None:
    async def broken(ctx, tool_name, arguments):
        return ToolResult(text="Error executing tool get_order: boom", is_error=True)

    monkeypatch.setattr(a, "call_mcp_tool", broken)
    event = await a.get_order(_ctx("x"), {"order_number": "ORD-20260118-001"})
    assert event.output == a.MANUAL_MCP_TEXT


@pytest.mark.asyncio
async def test_not_found_is_kept_as_business_answer(monkeypatch) -> None:
    async def not_found(ctx, tool_name, arguments):
        return ToolResult(text="找不到訂單編號 ORD-20990101-001", is_error=True)

    monkeypatch.setattr(a, "call_mcp_tool", not_found)
    event = await a.get_order(_ctx("x"), {"order_number": "ORD-20990101-001"})
    assert event.output == "找不到訂單編號 ORD-20990101-001"


async def test_model_error_callback_returns_manual_sentinel() -> None:
    response = await a._on_model_error(None, None, RuntimeError("429"))
    intent = a.Intent.model_validate_json(response.content.parts[0].text)
    assert intent.reason == "model_error"


def test_model_error_sentinel_routes_to_manual() -> None:
    event = _route("我要查詢訂單 ORD-20260118-001", {"intent": "unclear", "reason": "model_error"})
    assert event.actions.route == "manual"
    assert event.output == a.MANUAL_LLM_TEXT


def test_budget_exhaustion_goes_to_manual(monkeypatch) -> None:
    monkeypatch.setattr(a, "MAX_LLM_CALLS", 0)
    event = a.dispatch(_ctx("查詢訂單"), "查詢訂單")
    assert event.actions.route == "manual"
    assert event.output == a.MANUAL_LLM_TEXT


# ---------------------------------------------------------------------------
# Full graph, fake model, stubbed MCP: the 8-turn acceptance set
# ---------------------------------------------------------------------------

ACCEPTANCE = [
    ("我要查詢訂單，訂單編號：ORD-20260118-001", "TOOL get_order"),
    ("我要查詢客戶的訂單清單，客戶名稱：史塔克工業", "TOOL list_orders"),
    ("史塔克工業的訂單各狀態有幾筆？總金額多少？", "TOOL order_summary"),
    ("今天台北天氣如何？", "只能協助訂單查詢"),
    ("幫我複製報價單 QT-20260330-001", "報價單相關操作"),
    ("我要把 ORD-20260322-001 改成已出貨", "唯讀"),
    ("我要查詢訂單", "請提供訂單編號"),
    ("ORD-20260322-001", "TOOL get_order"),
]


async def _answer(runner: InMemoryRunner, message: str) -> str:
    events = await runner.run_debug(message, quiet=True)
    texts = [
        "".join(part.text or "" for part in event.content.parts).strip()
        for event in events
        if event.content and event.content.parts
    ]
    return [text for text in texts if text][-1]


@pytest.mark.asyncio
async def test_model_failure_ends_in_manual_reply(monkeypatch) -> None:
    from google.adk.models import LlmCapabilities
    from google.adk.models.base_llm import BaseLlm

    class _Boom(BaseLlm):
        model: str = "boom"

        @property
        def capabilities(self) -> LlmCapabilities:
            return LlmCapabilities(output_schema_and_tools=True)

        async def generate_content_async(self, request, stream: bool = False):
            raise RuntimeError("429 quota")
            yield  # pragma: no cover

    monkeypatch.setattr(a, "_build_model", lambda: _Boom())
    runner = InMemoryRunner(app=a.build_app(fake_llm=False))
    answer = await _answer(runner, "我要查詢訂單，訂單編號：ORD-20260118-001")
    assert answer == a.MANUAL_LLM_TEXT


@pytest.mark.asyncio
async def test_acceptance_set_through_the_graph(monkeypatch) -> None:
    calls: list[tuple[str, dict]] = []

    async def fake_mcp(ctx, tool_name, arguments):
        calls.append((tool_name, arguments))
        return ToolResult(text=f"TOOL {tool_name} {arguments}")

    monkeypatch.setattr(a, "call_mcp_tool", fake_mcp)
    runner = InMemoryRunner(app=a.build_app(fake_llm=True))

    for message, expected in ACCEPTANCE:
        answer = await _answer(runner, message)
        assert expected in answer, f"{message!r} -> {answer!r}"

    # Three tools, three calls; refusals never reach the MCP layer.
    assert [name for name, _ in calls] == ["get_order", "list_orders", "order_summary", "get_order"]

    # Reads are read-only: no tool call may change state. Only the four turns
    # that actually need interpretation cost a model call: the three guard
    # refusals and the bare order-number follow-up are handled deterministically.
    assert a.llm_calls() == 4
