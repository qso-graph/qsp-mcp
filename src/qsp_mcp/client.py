"""HTTP client for OpenAI-compatible LLM endpoints.

Handles chat completions with function calling support.
Non-streaming only for v1 (streaming buffering is post-v1).
"""

from __future__ import annotations

import json
from typing import Any

import httpx


class LLMClient:
    """Client for OpenAI-compatible chat completions API."""

    def __init__(
        self,
        endpoint: str,
        *,
        api_key: str | None = None,
        model: str = "default",
        max_tokens: int = 2048,
        temperature: float = 0.3,
        timeout: float = 120.0,
    ) -> None:
        self._endpoint = endpoint
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature

        headers: dict[str, str] = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        self._http = httpx.Client(
            headers=headers,
            timeout=httpx.Timeout(timeout, connect=10.0),
        )

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        *,
        temperature: float | None = None,
        tool_choice: str = "auto",
    ) -> dict[str, Any]:
        """Send a chat completion request.

        Returns the raw response dict from the LLM endpoint.
        """
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "max_tokens": self._max_tokens,
            "temperature": temperature if temperature is not None else self._temperature,
        }

        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice

        resp = self._http.post(self._endpoint, json=payload)
        resp.raise_for_status()

        return resp.json()

    def close(self) -> None:
        """Close the HTTP client."""
        self._http.close()

    def __enter__(self) -> LLMClient:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


def extract_tool_calls(response: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract tool calls from a chat completion response.

    Returns list of dicts with: id, name, arguments (parsed JSON).
    Returns empty list if no tool calls.
    """
    choices = response.get("choices", [])
    if not choices:
        return []

    message = choices[0].get("message", {})
    raw_calls = message.get("tool_calls", [])

    tool_calls = []
    for call in raw_calls:
        func = call.get("function", {})
        args_str = func.get("arguments", "{}")

        try:
            arguments = json.loads(args_str)
        except json.JSONDecodeError:
            arguments = {"_raw": args_str, "_parse_error": True}

        tool_calls.append(
            {
                "id": call.get("id", ""),
                "name": func.get("name", ""),
                "arguments": arguments,
            }
        )

    return tool_calls


def extract_text(response: dict[str, Any]) -> str:
    """Extract text content from a chat completion response."""
    choices = response.get("choices", [])
    if not choices:
        return ""

    message = choices[0].get("message", {})
    return message.get("content", "") or ""


def get_finish_reason(response: dict[str, Any]) -> str:
    """Get the finish reason from a response."""
    choices = response.get("choices", [])
    if not choices:
        return ""
    return choices[0].get("finish_reason", "")
