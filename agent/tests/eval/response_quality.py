"""Local LLM-as-judge for `custom_response_quality` (1-5)."""

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
    score: int  # 1-5
    explanation: str


def _client() -> genai.Client:
    """One client per grading thread (the eval SDK uses a thread pool)."""
    client = getattr(_local, "client", None)
    if client is None:
        client = _local.client = genai.Client(
            http_options=types.HttpOptions(
                retry_options=types.HttpRetryOptions(
                    initial_delay=2.0,
                    max_delay=60.0,
                    attempts=8,
                )
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


def evaluate(instance):
    reference = _text(instance.get("reference"))
    response_text = _text(instance.get("response"))
    agent_data = instance.get("agent_data")
    prompt_text = _text(instance.get("prompt"))

    rubric = (
        "Grade the agent's final response on a 1-5 scale (1 poor, 5 excellent) "
        "for accuracy, relevance, and clarity. For multi-turn traces, judge the "
        "final response against the final user turn and use the earlier turns "
        "only as context. Penalize Markdown links; the required format is a "
        "plain-text URL such as http://localhost:3000/quotations/12."
    )
    prompt = f"You are an expert QA evaluator for an enterprise AI assistant. {rubric}\n"
    if prompt_text:
        prompt += f"User Prompt: {prompt_text}\n"
    prompt += f"Final Response: {response_text}\n"
    if reference:
        prompt += f"Expected Answer (ground truth): {reference}\n"
    if agent_data:
        prompt += (
            "Full Agent Trace: "
            + json.dumps(agent_data, ensure_ascii=False, default=str)
            + "\n"
        )

    response = _client().models.generate_content(
        model=os.environ.get("AGENT_MODEL", "gemini-3.8-flash"),
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0,
            response_mime_type="application/json",
            response_schema=_Verdict,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )
    verdict = response.parsed
    if verdict is None:
        return {"score": 0, "explanation": response.text or ""}
    return {
        "score": max(1, min(5, verdict.score)),
        "explanation": verdict.explanation,
    }
