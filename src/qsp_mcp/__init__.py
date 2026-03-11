"""QSP-MCP — relay MCP tools to any OpenAI-compatible local LLM endpoint."""

from __future__ import annotations

try:
    from importlib.metadata import version

    __version__ = version("qsp-mcp")
except Exception:
    __version__ = "0.0.0-dev"
