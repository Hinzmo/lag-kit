"""The bridge: makes the harness an MCP client.

Connects to an MCP server over stdio, discovers its tools with tools/list,
translates each MCP tool schema into Ollama's tool-calling format, and routes the
model's tool_calls back to the server via tools/call. This is the layer that
decouples the agent loop from where tools physically live.

Wrinkle: the MCP SDK is async, but OllamaClient.run is sync. We reconcile that by
running one persistent asyncio loop in a daemon thread and marshalling each call
onto it with run_coroutine_threadsafe. That keeps the session (and the server
subprocess) alive across the whole run instead of standing it up per call.
"""
from __future__ import annotations
import asyncio
import threading

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class MCPBridge:
    """Owns a persistent MCP session on a background event loop."""

    def __init__(self, command: str, args: list[str]):
        self._loop = asyncio.new_event_loop()
        threading.Thread(target=self._spin, daemon=True).start()
        self.tools = []
        # block until the session is connected and tools are discovered
        self._submit(self._connect(command, args)).result(timeout=30)

    def _spin(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _submit(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    async def _connect(self, command, args):
        params = StdioServerParameters(command=command, args=args)
        # Enter the context managers manually so the session persists beyond this
        # coroutine. (A cleaner refactor later: keep the whole run inside the
        # `async with` blocks. This works and keeps the loop sync for now.)
        self._stdio_cm = stdio_client(params)
        read, write = await self._stdio_cm.__aenter__()
        self._session_cm = ClientSession(read, write)
        self.session = await self._session_cm.__aenter__()
        await self.session.initialize()
        self.tools = (await self.session.list_tools()).tools

    def call(self, name: str, args: dict) -> str:
        result = self._submit(self.session.call_tool(name, args or {})).result(timeout=60)
        # call_tool returns content blocks; pull text out of each
        return "\n".join(getattr(c, "text", str(c)) for c in result.content)


class MCPToolProvider:
    """Tool provider backed by an MCP server. Same interface as LocalToolProvider."""

    def __init__(self, bridge: MCPBridge):
        self.bridge = bridge

    @classmethod
    def from_config(cls, cfg: dict, server: str = "servers/tools_server.py"):
        return cls(MCPBridge("python", [server]))

    def schemas(self):
        # MCP gives us {name, description, inputSchema (JSON Schema)}.
        # Ollama wants {type: function, function: {name, description, parameters}}.
        # The translation is basically: inputSchema -> parameters.
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description or "",
                    "parameters": t.inputSchema,
                },
            }
            for t in self.bridge.tools
        ]

    def call(self, name: str, args: dict) -> str:
        return self.bridge.call(name, args)
