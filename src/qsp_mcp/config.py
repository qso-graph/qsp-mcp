"""Configuration loader for qsp-mcp.

Config format is Claude Desktop compatible — users can copy their existing
mcpServers block directly. The 'bridge' section is qsp-mcp specific.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ServerConfig:
    """MCP server configuration — matches Claude Desktop format."""

    name: str
    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)


@dataclass
class ProfileConfig:
    """Tool profile — subset of servers with model-specific params."""

    servers: list[str] | str  # list of server names or "*" for all
    temperature: float = 0.3
    system_prompt: str | None = None


@dataclass
class BridgeConfig:
    """Bridge-specific configuration."""

    endpoint: str = "http://localhost:8000/v1/chat/completions"
    api_key: str | None = None
    model: str = "default"
    max_tokens: int = 2048
    temperature: float = 0.3
    system_prompt: str = "You are an expert ham radio operator and RF engineer."
    max_history_turns: int = 5
    max_tool_calls_per_turn: int = 5
    tool_timeout_seconds: float = 5.0
    profile: str = "full"
    profiles: dict[str, ProfileConfig] = field(default_factory=dict)
    server_timeouts: dict[str, float] = field(default_factory=dict)


@dataclass
class Config:
    """Top-level qsp-mcp configuration."""

    servers: dict[str, ServerConfig] = field(default_factory=dict)
    bridge: BridgeConfig = field(default_factory=BridgeConfig)


def load_config(
    config_path: str | Path | None = None,
    *,
    endpoint: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
    profile: str | None = None,
    enable_writes: bool = False,
) -> Config:
    """Load config from file, with CLI overrides applied on top."""
    if config_path is not None:
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        raw = json.loads(path.read_text())
    else:
        raw = {}

    config = _parse_config(raw)

    # CLI overrides take precedence
    if endpoint is not None:
        config.bridge.endpoint = endpoint
    if api_key is not None:
        config.bridge.api_key = api_key
    if model is not None:
        config.bridge.model = model
    if profile is not None:
        config.bridge.profile = profile

    return config


def _parse_config(raw: dict[str, Any]) -> Config:
    """Parse raw JSON dict into typed Config."""
    config = Config()

    # Parse mcpServers (Claude Desktop compatible)
    for name, server_raw in raw.get("mcpServers", {}).items():
        config.servers[name] = ServerConfig(
            name=name,
            command=server_raw.get("command", ""),
            args=server_raw.get("args", []),
            env=server_raw.get("env", {}),
        )

    # Parse bridge section
    bridge_raw = raw.get("bridge", {})
    config.bridge = BridgeConfig(
        endpoint=bridge_raw.get("endpoint", config.bridge.endpoint),
        api_key=bridge_raw.get("api_key", config.bridge.api_key),
        model=bridge_raw.get("model", config.bridge.model),
        max_tokens=bridge_raw.get("max_tokens", config.bridge.max_tokens),
        temperature=bridge_raw.get("temperature", config.bridge.temperature),
        system_prompt=bridge_raw.get("system_prompt", config.bridge.system_prompt),
        max_history_turns=bridge_raw.get(
            "max_history_turns", config.bridge.max_history_turns
        ),
        max_tool_calls_per_turn=bridge_raw.get(
            "max_tool_calls_per_turn", config.bridge.max_tool_calls_per_turn
        ),
        tool_timeout_seconds=bridge_raw.get(
            "tool_timeout_seconds", config.bridge.tool_timeout_seconds
        ),
        profile=bridge_raw.get("profile", config.bridge.profile),
        server_timeouts={
            k: float(v) for k, v in bridge_raw.get("server_timeouts", {}).items()
        },
    )

    # Parse profiles
    for name, prof_raw in bridge_raw.get("profiles", {}).items():
        config.bridge.profiles[name] = ProfileConfig(
            servers=prof_raw.get("servers", "*"),
            temperature=prof_raw.get("temperature", config.bridge.temperature),
            system_prompt=prof_raw.get("system_prompt"),
        )

    # Ensure "full" profile always exists
    if "full" not in config.bridge.profiles:
        config.bridge.profiles["full"] = ProfileConfig(servers="*")

    return config


def get_api_key(config: Config) -> str | None:
    """Get API key from config."""
    return config.bridge.api_key
