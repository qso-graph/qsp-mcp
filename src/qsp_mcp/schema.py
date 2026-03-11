"""MCP tool schema → OpenAI tools format translator.

Converts MCP tool definitions into the OpenAI `tools` array format
(not legacy `functions` format) for consumption by any OpenAI-compatible
LLM endpoint.
"""

from __future__ import annotations

from typing import Any


def mcp_to_openai_tools(mcp_tools: list[Any]) -> list[dict[str, Any]]:
    """Convert MCP tool definitions to OpenAI tools array format.

    MCP tool format:
        name: str
        description: str
        inputSchema: dict (JSON Schema)

    OpenAI tools format:
        type: "function"
        function:
            name: str
            description: str
            parameters: dict (JSON Schema)
    """
    openai_tools = []
    for tool in mcp_tools:
        name = tool.name if hasattr(tool, "name") else tool.get("name", "")
        description = (
            tool.description
            if hasattr(tool, "description")
            else tool.get("description", "")
        )
        input_schema = (
            tool.inputSchema
            if hasattr(tool, "inputSchema")
            else tool.get("inputSchema", {})
        )

        # Clean the schema — remove keys that OpenAI doesn't expect
        parameters = _clean_schema(input_schema)

        openai_tools.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description or "",
                    "parameters": parameters,
                },
            }
        )

    return openai_tools


def _clean_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Clean JSON Schema for OpenAI compatibility.

    Removes fields that OpenAI's API doesn't accept and ensures
    required fields are present.
    """
    if not schema:
        return {"type": "object", "properties": {}}

    cleaned: dict[str, Any] = {}

    # Core schema fields
    cleaned["type"] = schema.get("type", "object")

    if "properties" in schema:
        cleaned["properties"] = {}
        for prop_name, prop_schema in schema["properties"].items():
            cleaned["properties"][prop_name] = _clean_property(prop_schema)

    if "required" in schema:
        cleaned["required"] = schema["required"]

    return cleaned


def _clean_property(prop: dict[str, Any]) -> dict[str, Any]:
    """Clean a single property schema."""
    cleaned: dict[str, Any] = {}

    for key in ("type", "description", "enum", "default", "items"):
        if key in prop:
            cleaned[key] = prop[key]

    return cleaned


def filter_tools_by_profile(
    tools: list[dict[str, Any]],
    tool_server_map: dict[str, str],
    allowed_servers: list[str] | str,
) -> list[dict[str, Any]]:
    """Filter OpenAI tools list to only include tools from allowed servers.

    Args:
        tools: OpenAI-format tools list.
        tool_server_map: Maps tool name → server name.
        allowed_servers: List of server names, or "*" for all.

    Returns:
        Filtered tools list.
    """
    if allowed_servers == "*":
        return tools

    return [
        tool
        for tool in tools
        if tool_server_map.get(tool["function"]["name"]) in allowed_servers
    ]


def filter_write_tools(
    tools: list[dict[str, Any]],
    enable_writes: bool,
) -> list[dict[str, Any]]:
    """Remove write-capable tools unless explicitly enabled.

    Write tools are identified by common naming patterns.
    """
    if enable_writes:
        return tools

    write_patterns = (
        "upload", "send", "post", "delete", "create", "update",
        "write", "modify", "remove", "set_",
    )

    return [
        tool
        for tool in tools
        if not any(
            tool["function"]["name"].lower().startswith(p)
            or f"_{p}" in tool["function"]["name"].lower()
            for p in write_patterns
        )
    ]
