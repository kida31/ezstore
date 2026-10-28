"""Integration check for the read-only order tools against the local MCP server.

Run with the Compose mcp-server exposed on localhost:8010:
    uv run pytest tests/integration/test_order_mcp.py -q

No Gemini calls are made here.
"""

from __future__ import annotations

import os
import uuid

import pytest
from google.adk.agents.invocation_context import InvocationContext
from google.adk.sessions import InMemorySessionService
from google.adk.tools.mcp_tool import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import (
    StreamableHTTPConnectionParams,
)
from google.adk.tools.tool_context import ToolContext

MCP_URL = os.getenv("MCP_URL", "http://localhost:8010/mcp")


async def _tool(name: str):
    toolset = McpToolset(
        connection_params=StreamableHTTPConnectionParams(url=MCP_URL)
    )
    tools = {tool.name: tool for tool in await toolset.get_tools()}
    return toolset, tools[name]


async def _call(name: str, args: dict) -> str:
    toolset, tool = await _tool(name)
    try:
        session_service = InMemorySessionService()
        session = await session_service.create_session(
            app_name="test", user_id="integration-user"
        )
        invocation = InvocationContext(
            session_service=session_service,
            invocation_id=str(uuid.uuid4()),
            session=session,
        )
        result = await tool.run_async(args=args, tool_context=ToolContext(invocation))
        return result["content"][0]["text"]
    finally:
        await toolset.close()


@pytest.mark.asyncio
async def test_get_order() -> None:
    try:
        text = await _call("get_order", {"order_number": "ORD-20260118-001"})
    except Exception as exc:  # noqa: BLE001 - skip when server is not running
        pytest.skip(f"MCP server is not reachable at {MCP_URL}: {exc}")

    assert "ORD-20260118-001" in text
    assert "已完成" in text
    assert "253350" in text
    assert "/orders/1" in text


@pytest.mark.asyncio
async def test_get_order_not_found() -> None:
    try:
        text = await _call("get_order", {"order_number": "ORD-20990101-001"})
    except Exception as exc:  # noqa: BLE001 - skip when server is not running
        pytest.skip(f"MCP server is not reachable at {MCP_URL}: {exc}")
    # ToolError surfaces as an MCP error payload; the message must be honest.
    assert "找不到訂單編號 ORD-20990101-001" in text


@pytest.mark.asyncio
async def test_list_orders_filters_customer() -> None:
    try:
        text = await _call("list_orders", {"customer_name": "史塔克工業"})
    except Exception as exc:  # noqa: BLE001 - skip when server is not running
        pytest.skip(f"MCP server is not reachable at {MCP_URL}: {exc}")

    assert "訂單清單" in text
    assert "/orders/" in text


@pytest.mark.asyncio
async def test_summary_and_status_whitelist() -> None:
    try:
        summary = await _call("order_summary", {})
        rejected = await _call("list_orders", {"status": "已出貨"})
    except Exception as exc:  # noqa: BLE001 - skip when server is not running
        pytest.skip(f"MCP server is not reachable at {MCP_URL}: {exc}")

    assert "訂單彙總" in summary
    assert "pending" in summary
    assert "不支援" in rejected
