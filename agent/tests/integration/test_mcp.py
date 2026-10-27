"""Integration check against the local MCP server.

Run with the Compose mcp-server exposed on localhost:8010:
    uv run pytest tests/integration/test_mcp.py -q
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


@pytest.mark.asyncio
async def test_get_quotation_via_mcp_toolset() -> None:
    toolset = McpToolset(
        connection_params=StreamableHTTPConnectionParams(url=MCP_URL)
    )
    try:
        try:
            tools = {tool.name: tool for tool in await toolset.get_tools()}
        except Exception as exc:  # noqa: BLE001 - skip when server is not running
            pytest.skip(f"MCP server is not reachable at {MCP_URL}: {exc}")

        assert "get_quotation" in tools

        session_service = InMemorySessionService()
        session = await session_service.create_session(
            app_name="test", user_id="integration-user"
        )
        invocation = InvocationContext(
            session_service=session_service,
            invocation_id=str(uuid.uuid4()),
            session=session,
        )
        tool_context = ToolContext(invocation)

        result = await tools["get_quotation"].run_async(
            args={"quotation_number": "QT-20260330-001"},
            tool_context=tool_context,
        )

        assert result["isError"] is False
        text = result["content"][0]["text"]
        assert "QT-20260330-001" in text
        assert "A4影印紙" in text
        assert "/quotations/12" in text
    finally:
        await toolset.close()
