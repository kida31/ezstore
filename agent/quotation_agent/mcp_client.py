"""ADK McpToolset integration for the provided quotation MCP server."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any

from google.adk.agents.context import Context
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.mcp_tool import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import (
    StreamableHTTPConnectionParams,
)
from google.adk.tools.tool_context import ToolContext

logger = logging.getLogger(__name__)

MCP_URL = os.getenv("MCP_URL", "http://localhost:8010/mcp")

_toolset: McpToolset | None = None
_tools: dict[str, BaseTool] | None = None
_tools_lock: asyncio.Lock | None = None
_tools_lock_loop: asyncio.AbstractEventLoop | None = None


@dataclass(slots=True)
class ToolResult:
    text: str
    is_error: bool = False
    raw: Any = None


def _get_toolset() -> McpToolset:
    global _toolset
    if _toolset is None:
        _toolset = McpToolset(
            connection_params=StreamableHTTPConnectionParams(
                url=MCP_URL,
                timeout=5.0,
                sse_read_timeout=300.0,
            )
        )
    return _toolset


def _get_tools_lock() -> asyncio.Lock:
    global _tools_lock, _tools_lock_loop
    loop = asyncio.get_running_loop()
    if _tools_lock is None or _tools_lock_loop is not loop:
        _tools_lock = asyncio.Lock()
        _tools_lock_loop = loop
    return _tools_lock


async def _get_tools() -> dict[str, BaseTool]:
    global _tools
    if _tools is not None:
        return _tools

    lock = _get_tools_lock()
    async with lock:
        if _tools is None:
            tools = await _get_toolset().get_tools()
            _tools = {tool.name: tool for tool in tools}
    return _tools


def _part_text(part: Any) -> str:
    if isinstance(part, dict):
        return str(part.get("text") or "")
    return str(getattr(part, "text", "") or "")


def parse_tool_result(tool_name: str, raw: Any) -> ToolResult:
    """Convert an ADK/MCP result into text while preserving the server message."""
    is_error = False
    text = ""

    if isinstance(raw, dict):
        is_error = bool(raw.get("isError") or raw.get("is_error"))
        content = raw.get("content") or []
        if isinstance(content, list):
            text = "\n".join(
                _part_text(part) for part in content if _part_text(part)
            ).strip()

        if not text:
            structured = raw.get("structuredContent") or raw.get("structured_content")
            if isinstance(structured, dict):
                text = str(structured.get("result") or "").strip()

        if not text and raw.get("error"):
            text = str(raw["error"]).strip()
            is_error = True
    else:
        text = str(raw or "").strip()

    # ADK wraps MCP tool failures as "Error executing tool <name>: <message>".
    # The user-facing answer should start with the useful message instead.
    text = re.sub(
        rf"^Error executing tool\s+{re.escape(tool_name)}\s*:\s*",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()
    text = re.sub(
        r"^Error executing tool\s*:\s*", "", text, flags=re.IGNORECASE
    ).strip()

    if not text:
        text = json.dumps(raw, ensure_ascii=False, default=str)
        is_error = True

    return ToolResult(text=text, is_error=is_error, raw=raw)


async def call_mcp_tool(
    ctx: Context,
    tool_name: str,
    arguments: dict[str, Any],
) -> ToolResult:
    """Call one MCP tool from a Workflow function node."""
    try:
        tools = await _get_tools()
    except Exception as exc:  # noqa: BLE001 - surfaced as a user-facing message
        logger.exception("Unable to load MCP tools from %s", MCP_URL)
        return ToolResult(
            text=f"操作失敗：無法連線報價單系統（{exc}）。",
            is_error=True,
        )

    tool = tools.get(tool_name)
    if tool is None:
        return ToolResult(
            text=f"操作失敗：報價單系統沒有提供 {tool_name} 工具。",
            is_error=True,
        )

    try:
        # A Workflow function node receives a Context, while BaseTool.run_async
        # expects a ToolContext. The MCP tool only needs the invocation context.
        tool_context = ToolContext(ctx._invocation_context)
        raw = await tool.run_async(args=arguments, tool_context=tool_context)
        return parse_tool_result(tool_name, raw)
    except Exception as exc:  # noqa: BLE001 - surfaced as a user-facing message
        logger.exception("MCP tool %s failed", tool_name)
        return ToolResult(
            text=f"操作失敗：呼叫 {tool_name} 時發生錯誤（{exc}）。",
            is_error=True,
        )


async def close() -> None:
    global _toolset, _tools, _tools_lock, _tools_lock_loop
    if _toolset is not None:
        try:
            await _toolset.close()
        except Exception:  # noqa: BLE001 - shutdown is best effort
            logger.debug("Ignoring MCP toolset close error", exc_info=True)
    _toolset = None
    _tools = None
    _tools_lock = None
    _tools_lock_loop = None
