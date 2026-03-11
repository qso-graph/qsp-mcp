"""CLI entry point for qsp-mcp.

Supports two modes:
- Interactive: readline-enabled chat loop
- Single query: --query "..." for one-shot usage
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from . import __version__
from .config import load_config
from .relay import QSPRelay

DEFAULT_CONFIG_PATHS = [
    Path.home() / ".config" / "qsp-mcp" / "config.json",
    Path.home() / ".qsp-mcp.json",
]


def _find_config() -> Path | None:
    """Find the default config file."""
    for path in DEFAULT_CONFIG_PATHS:
        if path.exists():
            return path
    return None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="qsp-mcp",
        description="QSP — relay MCP tools to any OpenAI-compatible LLM endpoint",
    )
    parser.add_argument(
        "--version", action="version", version=f"qsp-mcp {__version__}"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="Path to config file (default: ~/.config/qsp-mcp/config.json)",
    )
    parser.add_argument(
        "--endpoint", "-e",
        type=str,
        default=None,
        help="LLM endpoint URL (overrides config)",
    )
    parser.add_argument(
        "--api-key", "-k",
        type=str,
        default=None,
        help="API key for the LLM endpoint",
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help="Model name (overrides config)",
    )
    parser.add_argument(
        "--profile", "-p",
        type=str,
        default=None,
        help="Tool profile to use (e.g., contest, dx, propagation, full)",
    )
    parser.add_argument(
        "--query", "-q",
        type=str,
        default=None,
        help="Single query mode — ask one question and exit",
    )
    parser.add_argument(
        "--enable-writes",
        action="store_true",
        default=False,
        help="Enable write-capable tools (disabled by default for safety)",
    )
    parser.add_argument(
        "--list-tools",
        action="store_true",
        default=False,
        help="List available tools and exit",
    )

    return parser.parse_args()


async def _run(args: argparse.Namespace) -> int:
    """Main async entry point."""
    # Find config
    config_path = args.config
    if config_path is None:
        found = _find_config()
        if found:
            config_path = str(found)

    if config_path is None and args.endpoint is None:
        print(
            "Error: No config file found and no --endpoint specified.\n"
            "Create ~/.config/qsp-mcp/config.json or use --endpoint.\n"
            "See: https://github.com/qso-graph/qsp-mcp",
            file=sys.stderr,
        )
        return 1

    try:
        config = load_config(
            config_path,
            endpoint=args.endpoint,
            api_key=args.api_key,
            model=args.model,
            profile=args.profile,
            enable_writes=args.enable_writes,
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"Config error: {e}", file=sys.stderr)
        return 1

    if not config.servers:
        print(
            "Error: No MCP servers configured in mcpServers block.",
            file=sys.stderr,
        )
        return 1

    relay = QSPRelay(config, enable_writes=args.enable_writes)

    try:
        await relay.start()
    except RuntimeError as e:
        print(f"Startup failed: {e}", file=sys.stderr)
        return 1

    try:
        # List tools mode
        if args.list_tools:
            print(relay.get_tool_summary())
            for tool in relay._openai_tools:
                func = tool["function"]
                server = relay._tool_server_map.get(func["name"], "?")
                print(f"  [{server}] {func['name']}: {func['description'][:80]}")
            return 0

        # Single query mode
        if args.query:
            response = await relay.query(args.query, [])
            print(response)
            return 0

        # Interactive mode
        return await _interactive_loop(relay, config.bridge.max_history_turns)

    finally:
        await relay.stop()


async def _interactive_loop(relay: QSPRelay, max_history: int) -> int:
    """Interactive chat loop with readline support."""
    history: list[dict[str, str]] = []

    print(
        f"qsp-mcp {__version__} — type your question, or 'quit' to exit.\n"
        f"Servers:\n{relay.get_tool_summary()}\n"
    )

    while True:
        try:
            user_input = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n73!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q", "73"):
            print("73!")
            break

        if user_input.lower() == "/tools":
            print(relay.get_tool_summary())
            for tool in relay._openai_tools:
                func = tool["function"]
                server = relay._tool_server_map.get(func["name"], "?")
                print(f"  [{server}] {func['name']}")
            continue

        if user_input.lower() == "/help":
            print(
                "Commands:\n"
                "  /tools  — list available tools\n"
                "  /help   — show this help\n"
                "  quit    — exit (or 73)\n"
            )
            continue

        try:
            response = await relay.query(user_input, history)
        except Exception as e:
            print(f"[qsp] Error: {type(e).__name__}: {e}", file=sys.stderr)
            continue

        print(f"\n{response}\n")

        # Update sliding window history
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": response})

        # Trim to max_history turns (each turn = user + assistant = 2 messages)
        max_messages = max_history * 2
        if len(history) > max_messages:
            history = history[-max_messages:]

    return 0


def main() -> None:
    """Entry point for the qsp-mcp CLI."""
    try:
        exit_code = asyncio.run(_run(_parse_args()))
    except KeyboardInterrupt:
        print("\n73!")
        exit_code = 0
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
