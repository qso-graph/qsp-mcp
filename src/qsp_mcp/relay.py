"""QSP Relay — the stateless pipe between LLM and MCP tools.

This is the core of qsp-mcp. It manages:
- MCP server lifecycle (start/stop via stdio)
- Tool discovery and schema translation
- Conversation loop with tool call/result cycling
- Tool validation and routing

Design: strict stateless pipe. No caching, no shared state, no health polling.
All state lives in MCP servers. All inference optimization lives in the
inference server. QSP just connects the two sides.
"""

from __future__ import annotations

import asyncio
import json
import sys
from contextlib import AsyncExitStack
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from .client import LLMClient, extract_text, extract_tool_calls, get_finish_reason
from .config import BridgeConfig, Config, ProfileConfig, ServerConfig, get_api_key
from .schema import (
    filter_tools_by_profile,
    filter_write_tools,
    mcp_to_openai_tools,
)


class MCPServerHandle:
    """Handle to a running MCP server process."""

    def __init__(self, name: str, session: ClientSession) -> None:
        self.name = name
        self.session = session
        self.tools: list[Any] = []
        self.available = True


class QSPRelay:
    """The QSP relay — connects MCP tools to an LLM endpoint."""

    def __init__(
        self,
        config: Config,
        *,
        enable_writes: bool = False,
    ) -> None:
        self._config = config
        self._enable_writes = enable_writes
        self._servers: dict[str, MCPServerHandle] = {}
        self._tool_server_map: dict[str, str] = {}  # tool_name → server_name
        self._openai_tools: list[dict[str, Any]] = []
        self._exit_stack = AsyncExitStack()
        self._llm: LLMClient | None = None

    async def start(self) -> None:
        """Start all configured MCP servers and discover tools."""
        await self._exit_stack.__aenter__()

        # Start MCP servers
        for name, server_cfg in self._config.servers.items():
            try:
                await self._start_server(name, server_cfg)
            except Exception as e:
                print(f"[qsp] Failed to start {name}: {e}", file=sys.stderr)

        if not self._servers:
            raise RuntimeError("No MCP servers started successfully")

        # Discover tools from all servers
        await self._discover_tools()

        # Apply profile filtering
        profile = self._get_active_profile()
        if profile:
            self._openai_tools = filter_tools_by_profile(
                self._openai_tools,
                self._tool_server_map,
                profile.servers,
            )

        # Apply write filtering
        self._openai_tools = filter_write_tools(
            self._openai_tools, self._enable_writes
        )

        # Initialize LLM client
        api_key = get_api_key(self._config)
        self._llm = LLMClient(
            self._config.bridge.endpoint,
            api_key=api_key,
            model=self._config.bridge.model,
            max_tokens=self._config.bridge.max_tokens,
            temperature=self._config.bridge.temperature,
        )

        server_count = len(self._servers)
        tool_count = len(self._openai_tools)
        print(
            f"[qsp] Ready — {server_count} server(s), {tool_count} tool(s)",
            file=sys.stderr,
        )

    async def stop(self) -> None:
        """Stop all MCP servers and clean up."""
        if self._llm:
            self._llm.close()
            self._llm = None
        await self._exit_stack.aclose()
        self._servers.clear()

    async def query(self, user_input: str, history: list[dict[str, Any]]) -> str:
        """Process a user query through the relay.

        Args:
            user_input: The user's message.
            history: Conversation history (sliding window).

        Returns:
            The LLM's final text response.
        """
        if not self._llm:
            raise RuntimeError("Relay not started — call start() first")

        bridge = self._config.bridge
        profile = self._get_active_profile()
        temperature = profile.temperature if profile else bridge.temperature
        system_prompt = (
            profile.system_prompt if profile and profile.system_prompt
            else bridge.system_prompt
        )

        # Build messages: system + history + new user message
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
        ]
        messages.extend(history)
        messages.append({"role": "user", "content": user_input})

        # Tool call loop — up to max_tool_calls_per_turn iterations
        tool_calls_this_turn = 0

        while True:
            response = self._llm.chat(
                messages,
                tools=self._openai_tools if self._openai_tools else None,
                temperature=temperature,
            )

            finish_reason = get_finish_reason(response)
            tool_calls = extract_tool_calls(response)

            # No tool calls — return the text response
            if not tool_calls or finish_reason != "tool_calls":
                text = extract_text(response)
                return text

            # Depth cap check
            tool_calls_this_turn += len(tool_calls)
            if tool_calls_this_turn > bridge.max_tool_calls_per_turn:
                text = extract_text(response)
                if text:
                    return text
                return (
                    f"[qsp] Tool call limit reached "
                    f"({bridge.max_tool_calls_per_turn} per turn). "
                    f"Partial results returned."
                )

            # Add assistant message with tool calls to conversation
            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": json.dumps(tc["arguments"]),
                        },
                    }
                    for tc in tool_calls
                ],
            }
            messages.append(assistant_msg)

            # Execute each tool call and add results
            for tc in tool_calls:
                result = await self._execute_tool(tc["name"], tc["arguments"])
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": result,
                    }
                )

    async def _start_server(
        self, name: str, server_cfg: ServerConfig
    ) -> None:
        """Start a single MCP server via stdio."""
        params = StdioServerParameters(
            command=server_cfg.command,
            args=server_cfg.args,
            env=server_cfg.env if server_cfg.env else None,
        )

        stdio_transport = await self._exit_stack.enter_async_context(
            stdio_client(params)
        )
        read_stream, write_stream = stdio_transport
        session = await self._exit_stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )
        await session.initialize()

        self._servers[name] = MCPServerHandle(name=name, session=session)

    async def _discover_tools(self) -> None:
        """Discover tools from all running MCP servers."""
        all_mcp_tools = []

        for name, handle in self._servers.items():
            try:
                result = await handle.session.list_tools()
                handle.tools = result.tools
                for tool in result.tools:
                    tool_name = tool.name
                    self._tool_server_map[tool_name] = name
                    all_mcp_tools.append(tool)
            except Exception as e:
                print(
                    f"[qsp] Failed to list tools from {name}: {e}",
                    file=sys.stderr,
                )
                handle.available = False

        self._openai_tools = mcp_to_openai_tools(all_mcp_tools)

    async def _execute_tool(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> str:
        """Execute a tool call against the appropriate MCP server.

        Returns the result as a string (for inclusion in conversation).
        """
        server_name = self._tool_server_map.get(tool_name)
        if not server_name:
            return json.dumps({"error": f"Unknown tool: {tool_name}"})

        handle = self._servers.get(server_name)
        if not handle or not handle.available:
            return json.dumps(
                {"error": f"Server {server_name} unavailable for tool {tool_name}"}
            )

        # Get timeout for this server
        timeout = self._config.bridge.server_timeouts.get(
            server_name, self._config.bridge.tool_timeout_seconds
        )

        try:
            result = await asyncio.wait_for(
                handle.session.call_tool(tool_name, arguments),
                timeout=timeout,
            )

            # Extract text content from MCP result
            if result.content:
                parts = []
                for block in result.content:
                    if hasattr(block, "text"):
                        parts.append(block.text)
                    else:
                        parts.append(str(block))
                return "\n".join(parts)

            return json.dumps({"result": "Tool returned no content"})

        except asyncio.TimeoutError:
            return json.dumps(
                {
                    "error": f"Tool {tool_name} timed out after {timeout}s",
                    "server": server_name,
                }
            )
        except ConnectionError:
            handle.available = False
            return json.dumps(
                {
                    "error": f"Server {server_name} connection lost",
                    "tool": tool_name,
                }
            )
        except Exception as e:
            return json.dumps(
                {
                    "error": f"Tool execution failed: {type(e).__name__}",
                    "tool": tool_name,
                }
            )

    def _get_active_profile(self) -> ProfileConfig | None:
        """Get the active profile config."""
        name = self._config.bridge.profile
        return self._config.bridge.profiles.get(name)

    def get_tool_summary(self) -> str:
        """Return a human-readable summary of available tools."""
        lines = []
        for name, handle in self._servers.items():
            status = "ok" if handle.available else "DOWN"
            tool_count = len(
                [t for t in self._openai_tools
                 if self._tool_server_map.get(t["function"]["name"]) == name]
            )
            lines.append(f"  {name}: {tool_count} tools [{status}]")
        return "\n".join(lines)
