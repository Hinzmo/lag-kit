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
import sys
import threading
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# repo root: src/lag/mcp_client.py -> src/lag -> src -> root. Server scripts live
# under servers/, so a bare relative path must not depend on the caller's cwd.
_REPO_ROOT = Path(__file__).resolve().parents[2]


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


def _input_schema(tool) -> dict:
    """Return a tool's JSON Schema across SDK versions.

    mcp 1.x exposes it as `inputSchema`; mcp 2.x renamed it to `input_schema`.
    Both are plain JSON Schema, so only the attribute name differs.
    """
    schema = getattr(tool, "input_schema", None)
    if schema is None:
        schema = getattr(tool, "inputSchema", None)
    return schema or {"type": "object", "properties": {}}


class MCPToolProvider:
    """Tool provider backed by one or more MCP servers. Same interface as LocalToolProvider.

    Holds a *list* of bridges so a single agent loop can reach tools spread across several
    servers — e.g. tools_server.py (fs, shell, search) plus rag_server.py
    (search_knowledge_base) — without knowing they live in different processes. `schemas()`
    merges every server's tools into one flat list; `call()` routes each invocation to the
    bridge that owns the named tool. The loop still sees the same duck-typed (schemas, call)
    surface as the single-server and in-process providers, so nothing above this layer changes.
    """

    #: Servers from_config spawns, in precedence order (earlier wins a name collision).
    DEFAULT_SERVERS = ("servers/tools_server.py", "servers/rag_server.py")

    def __init__(self, bridges):
        # Accept a single bridge or an iterable of them, normalized to a list. A bridge is a
        # lone object; a list/tuple is several. Keeping single-bridge construction working
        # means existing callers and tests — MCPToolProvider(one_bridge) — are unchanged.
        self.bridges = list(bridges) if isinstance(bridges, (list, tuple)) else [bridges]

    @classmethod
    def from_config(cls, cfg: dict, servers: tuple[str, ...] | None = None):
        # sys.executable, not "python": the bare name is absent on plenty of distros (Ubuntu
        # ships only python3), and even where it resolves it is the *system* interpreter,
        # which cannot import lag. A server that dies on import leaves stdio_client waiting on
        # a pipe that never speaks — which is what the "--mcp hangs" symptom actually was.
        bridges = []
        for server in (servers or cls.DEFAULT_SERVERS):
            path = Path(server)
            if not path.is_absolute():
                path = _REPO_ROOT / path
            bridges.append(MCPBridge(sys.executable, [str(path)]))
        return cls(bridges)

    def _tool_owner(self, name: str):
        """Return the first bridge exposing `name`, or None. First bridge wins a collision."""
        for bridge in self.bridges:
            if any(getattr(t, "name", None) == name for t in bridge.tools):
                return bridge
        return None

    def schemas(self):
        # MCP gives us {name, description, input schema (JSON Schema)}; Ollama wants
        # {type: function, function: {name, description, parameters}}. The translation is
        # basically input schema -> parameters. Tools from every bridge are merged into one
        # flat list; a name seen twice is emitted once (first server wins, matching call()).
        seen: set[str] = set()
        out = []
        for bridge in self.bridges:
            for t in bridge.tools:
                if t.name in seen:
                    continue
                seen.add(t.name)
                out.append({
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description or "",
                        "parameters": _input_schema(t),
                    },
                })
        return out

    def call(self, name: str, args: dict) -> str:
        owner = self._tool_owner(name)
        if owner is None:
            # One bridge: delegate anyway — it's the only place the call can go, and this
            # keeps behaviour identical to the pre-merge single-server provider. Several
            # bridges: an unroutable name is a real error, returned as a string (never
            # raised) so the agent loop can recover.
            if len(self.bridges) == 1:
                owner = self.bridges[0]
            else:
                return f"unknown tool: {name}"
        return owner.call(name, args)
