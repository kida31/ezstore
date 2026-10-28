"""EZStore quotation management agent.

The graph deliberately has exactly one LLM node: `classify_intent`. Everything
else - tool selection, argument passing, clarification, refusal and the final
response - is deterministic Python.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

from google.adk.agents import LlmAgent
from google.adk.agents.context import Context
from google.adk.apps import App
from google.adk.events.event import Event
from google.adk.models import Gemini
from google.adk.workflow import Workflow
from google.genai import types

from quotation_agent import prompts, routing
from quotation_agent.mcp_client import call_mcp_tool
from quotation_agent.models import Intent, IntentType

try:
    from dotenv import load_dotenv

    # `adk web`/`agents-cli playground` imports the package directly; load the
    # project-local .env here as well as in fast_api_app.py.
    load_dotenv()
except ImportError:  # pragma: no cover - dotenv ships with google-adk
    pass


_MODEL_NAME = os.getenv("AGENT_MODEL", "gemini-3.8-flash")


def _env_true(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _thinking_config(model_name: str) -> types.ThinkingConfig | None:
    """Return the classifier's thinking config for the selected model.

    Lite models reject ``thinking_budget=0`` with HTTP 400, so omit the
    setting there and keep thinking disabled only on models that support it.
    """
    if "lite" in model_name.lower():
        return None
    return types.ThinkingConfig(thinking_budget=0)


def _build_model() -> Gemini:
    model_name = _MODEL_NAME
    use_vertex = _env_true("GOOGLE_GENAI_USE_VERTEXAI")
    # The scaffold historically used GOOGLE_GENAI_USE_ENTERPRISE. Respect it only
    # when no AI Studio key is present, so an AI Studio .env cannot be accidentally
    # switched to Vertex by a leftover template value.
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
        model=model_name,
        client_kwargs={
            **(client_kwargs or {}),
            # Cap every request at 10s so a stalled attempt cannot hang the
            # graph. google-genai applies this as a per-read timeout.
            "http_options": types.HttpOptions(timeout=10_000),
        },
        # Free-tier AI Studio keys allow only 5 requests/minute per model, and
        # some models (e.g. gemini-3.8-flash) answer 503 high-demand for long
        # stretches. Retry sparingly so one prompt cannot stall for minutes:
        # 10 + 0.5 + 10 + 1 + 10 ≈ 31.5s worst case over 3 attempts.
        retry_options=types.HttpRetryOptions(
            initial_delay=0.5,
            max_delay=1.0,
            attempts=3,
            jitter=0.0,
        ),
    )


classifier = LlmAgent(
    name="classify_intent",
    model=_build_model(),
    instruction=prompts.CLASSIFIER_INSTRUCTION,
    output_schema=Intent,
    generate_content_config=types.GenerateContentConfig(
        temperature=0.0,
        # Intent classification is a short structured task. Disable thinking so
        # the graph does not spend tens of seconds (or minutes) on hidden tokens.
        # Lite models reject thinking_budget=0, so `_thinking_config` omits it.
        thinking_config=_thinking_config(_MODEL_NAME),
    ),
)


def _content_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, dict):
        parts = content.get("parts") or []
    else:
        parts = getattr(content, "parts", None) or []
    chunks: list[str] = []
    for part in parts:
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


def _infer_pending_from_history(ctx: Context) -> Intent | None:
    """Recover a pending operation from prior events when state is unavailable.

    `agents-cli eval` seeds multi-turn trajectories as events. State deltas from
    the previous turn are not necessarily replayed, so inspect the most recent
    clarification and the user message before it.
    """
    events = list(getattr(ctx.session, "events", []) or [])
    clarify_index: int | None = None
    for index in range(len(events) - 1, -1, -1):
        if routing.is_clarification_text(_content_text(events[index].content)):
            clarify_index = index
            break
    if clarify_index is None:
        return None

    for index in range(clarify_index - 1, -1, -1):
        event = events[index]
        author = getattr(event, "author", "")
        if author == "user":
            return routing.infer_pending_from_user_text(
                _content_text(event.content)
            )
    return None


def _pending_as_intent(value: Any) -> Intent | None:
    if value is None:
        return None
    try:
        return Intent.model_validate(value)
    except Exception:  # noqa: BLE001 - stale/corrupt state is ignored
        return None


def _as_intent(value: Any) -> Intent:
    if isinstance(value, Intent):
        return value
    return Intent.model_validate(value)


def load_context(ctx: Context, node_input: Any) -> Event:
    """Read the current message and any pending operation from the session."""
    message = _user_text(ctx, node_input)
    pending = _pending_as_intent(ctx.state.get("pending_intent"))
    if pending is None:
        pending = _infer_pending_from_history(ctx)

    pending_payload = pending.model_dump(mode="json") if pending else None
    pending_summary = (
        json.dumps(pending_payload, ensure_ascii=False, default=str)
        if pending_payload
        else "（無）"
    )
    return Event(
        output=message,
        state={
            "current_message": message,
            "pending_intent": pending_payload,
            "pending_summary": pending_summary,
        },
    )


def guard(ctx: Context, node_input: str) -> Event:
    """Deterministically flag unmistakable out-of-scope requests."""
    reason = routing.deterministic_off_topic(node_input)
    return Event(
        output=node_input,
        state={
            "guard_off_topic": bool(reason),
            "guard_reason": reason or "",
        },
    )


def route_intent(ctx: Context, node_input: dict[str, Any]) -> Event:
    """Map a structured classifier result to one graph route."""
    message = str(ctx.state.get("current_message") or "")
    if ctx.state.get("guard_off_topic"):
        return Event(
            output={"reason": ctx.state.get("guard_reason") or "off_topic"},
            route=IntentType.OFF_TOPIC.value,
            state={"pending_intent": None},
        )

    try:
        classified = _as_intent(node_input)
    except Exception:  # noqa: BLE001 - malformed LLM output must not crash the graph
        classified = Intent(
            intent=IntentType.UNCLEAR,
            reason="classifier output could not be parsed",
        )

    # Safety net for short, unambiguous requests such as "我要查詢報價單".
    # The missing quotation number is a clarification field, not a reason to
    # classify the whole request as unclear.
    if classified.intent in (IntentType.OFF_TOPIC, IntentType.UNCLEAR):
        inferred = routing.infer_pending_from_user_text(message)
        if inferred is not None:
            classified = inferred

    pending = _pending_as_intent(ctx.state.get("pending_intent"))
    merged = routing.merge_intent(classified, pending, message)

    if merged.intent == IntentType.OFF_TOPIC:
        return Event(
            output={"reason": merged.reason or "off_topic"},
            route=IntentType.OFF_TOPIC.value,
            state={"pending_intent": None},
        )

    if merged.intent == IntentType.UNCLEAR:
        return Event(
            output={"intent": IntentType.UNCLEAR.value, "missing": [], "pending": None},
            route="clarify",
            state={"pending_intent": None},
        )

    missing = routing.missing_fields(merged)
    if missing:
        pending_payload = merged.model_dump(mode="json")
        return Event(
            output={
                "intent": merged.intent.value,
                "missing": missing,
                "pending": pending_payload,
            },
            route="clarify",
            state={"pending_intent": pending_payload},
        )

    return Event(
        output=merged.model_dump(mode="json"),
        route=merged.intent.value,
        state={"pending_intent": None},
    )


async def _run_operation(
    ctx: Context,
    tool_name: str,
    arguments: dict[str, Any],
    *,
    unchanged_note: str = "",
) -> Event:
    result = await call_mcp_tool(ctx, tool_name, arguments)
    text = result.text
    # Some operations (e.g. duplicate) never touch the source quotation, but the
    # tool text does not always say so. Add the reassurance when it is missing.
    if unchanged_note and "未更動" not in text and "未修改" not in text:
        text = f"{text}\n{unchanged_note}" if text else unchanged_note
    return Event(
        output=text,
        state={
            "pending_intent": None,
            "last_tool_error": result.is_error,
        },
    )


async def get_quotation(ctx: Context, node_input: dict[str, Any]) -> Event:
    request = _as_intent(node_input)
    return await _run_operation(
        ctx,
        "get_quotation",
        {"quotation_number": request.quotation_number},
    )


async def list_customer_quotations(
    ctx: Context, node_input: dict[str, Any]
) -> Event:
    request = _as_intent(node_input)
    return await _run_operation(
        ctx,
        "list_customer_quotations",
        {"customer_name": request.customer_name},
    )


async def duplicate_quotation(ctx: Context, node_input: dict[str, Any]) -> Event:
    request = _as_intent(node_input)
    return await _run_operation(
        ctx,
        "duplicate_quotation",
        {"quotation_number": request.quotation_number},
        unchanged_note="原始報價單未被更動。",
    )


async def update_pricing_factor(
    ctx: Context, node_input: dict[str, Any]
) -> Event:
    request = _as_intent(node_input)
    return await _run_operation(
        ctx,
        "update_pricing_factor",
        {
            "quotation_number": request.quotation_number,
            "new_pricing_factor": request.new_pricing_factor,
        },
    )


async def adjust_total_price(ctx: Context, node_input: dict[str, Any]) -> Event:
    request = _as_intent(node_input)
    return await _run_operation(
        ctx,
        "adjust_total_price",
        {
            "quotation_number": request.quotation_number,
            "target_total_price": request.target_total_price,
        },
    )


async def convert_to_order(ctx: Context, node_input: dict[str, Any]) -> Event:
    request = _as_intent(node_input)
    return await _run_operation(
        ctx,
        "convert_to_order",
        {"quotation_number": request.quotation_number},
    )


async def clarify(ctx: Context, node_input: dict[str, Any]) -> Event:
    text = routing.clarify_message(
        node_input.get("intent"),
        list(node_input.get("missing") or []),
    )
    return Event(
        output=text,
        state={"pending_intent": node_input.get("pending")},
    )


async def refuse(ctx: Context, node_input: dict[str, Any]) -> Event:
    return Event(
        output=prompts.REFUSAL_TEXT,
        state={"pending_intent": None, "guard_off_topic": False},
    )


_FIELD_LINE = re.compile(r"^([^\s:：][^:：]{0,23})[:：]\s*(.*)$")
_NUMBERED_ITEM = re.compile(r"^\s+\d+\.\s+(.*)$")


def _wants_markdown(message: str) -> bool:
    return re.search(r"markdown|\bmd\b", message, re.IGNORECASE) is not None


def _to_markdown(text: str) -> str:
    """Render a plain-text tool result as Markdown without making links Markdown.

    The spec requires links to stay as bare URLs, so only headings, bold field
    labels and list items are added; URLs are never wrapped in ``[text](url)``.
    """
    lines: list[str] = []
    for line in text.splitlines():
        field = _FIELD_LINE.match(line)
        if field:
            key = field.group(1).strip()
            value = field.group(2).strip()
            if value.startswith(("http://", "https://")):
                label = f"**{key}（純文字）**"
            else:
                label = f"**{key}**"
            lines.append(f"{label}：{value}" if value else label)
            continue
        item = _NUMBERED_ITEM.match(line)
        if item:
            lines.append(f"- {item.group(1).strip()}")
            continue
        lines.append(line)
    intro = "> 已依您的要求排版如下；系統連結依規範維持純文字。\n\n"
    return intro + "## 操作結果\n\n" + "\n".join(lines)


async def respond(ctx: Context, node_input: Any) -> Event:
    """Emit the only user-facing text event, under the fixed node name `respond`."""
    if isinstance(node_input, str):
        text = node_input.strip()
    elif node_input is None:
        text = ""
    else:
        text = json.dumps(node_input, ensure_ascii=False, default=str)
    if not text:
        text = "操作完成，但系統沒有回傳內容。"

    message = str(ctx.state.get("current_message") or "")
    if _wants_markdown(message) and "http" in text:
        text = _to_markdown(text)

    history = list(ctx.state.get("chat_history") or [])
    history.append(
        {
            "role": "user",
            "text": str(ctx.state.get("current_message") or ""),
        }
    )
    history.append({"role": "assistant", "text": text})
    content = types.Content(
        role="model",
        parts=[types.Part.from_text(text=text)],
    )
    return Event(
        output=text,
        content=content,
        state={
            "chat_history": history[-20:],
            "last_response": text,
        },
    )


edges = [
    ("START", load_context),
    (load_context, guard),
    (guard, classifier),
    (classifier, route_intent),
    (
        route_intent,
        {
            IntentType.GET_QUOTATION.value: get_quotation,
            IntentType.LIST_CUSTOMER_QUOTATIONS.value: list_customer_quotations,
            IntentType.DUPLICATE_QUOTATION.value: duplicate_quotation,
            IntentType.UPDATE_PRICING_FACTOR.value: update_pricing_factor,
            IntentType.ADJUST_TOTAL_PRICE.value: adjust_total_price,
            IntentType.CONVERT_TO_ORDER.value: convert_to_order,
            "clarify": clarify,
            IntentType.OFF_TOPIC.value: refuse,
        },
    ),
    (get_quotation, respond),
    (list_customer_quotations, respond),
    (duplicate_quotation, respond),
    (update_pricing_factor, respond),
    (adjust_total_price, respond),
    (convert_to_order, respond),
    (clarify, respond),
    (refuse, respond),
]

root_agent = Workflow(
    name="quotation_agent",
    description=(
        "EZStore quotation management agent: query, duplicate, reprice and "
        "convert quotations through the MCP server."
    ),
    edges=edges,
)

app = App(
    name="quotation_agent",
    root_agent=root_agent,
)
