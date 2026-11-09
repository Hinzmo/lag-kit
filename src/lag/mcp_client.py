"""The bridge: makes the harness an MCP client.

Connects to an MCP server over stdio, discovers its tools with tools/list,
translates each MCP tool schema into Ollama's tool-calling format, and routes the
model's tool_calls back to the server via tools/call. This is the layer that
decouples the agent loop from where tools physically live.

Wrinkle: the MCP SDK is async, but OllamaClient.run is sync. We reconcile that by
running one persistent asyncio loop in a daemon thread and marshalling each call
onto it with run_coroutine_threadsafe. That keeps the session (and the server
subprocess) alive across the whole run instead of standing it up per call. The
session's lifetime is a single task on that loop, so it is torn down by the task that
set it up — see MCPBridge.
"""
from __future__ import annotations
import asyncio
import concurrent.futures
import contextlib
import sys
import threading
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# repo root: src/lag/mcp_client.py -> src/lag -> src -> root. Server scripts live
# under servers/, so a bare relative path must not depend on the caller's cwd.
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _unwrap(e: BaseException) -> BaseException:
    """The one real error inside anyio's task-group wrapping.

    Raised from inside the SDK's `async with` blocks, a failure arrives as
    ExceptionGroup("unhandled errors in a TaskGroup (1 sub-exception)") — which hides
    the "MCPError: Connection closed" a person needs to see to know a server died on
    startup. Only single-error groups are unwrapped; a real group stays a group.
    """
    while isinstance(e, BaseExceptionGroup) and len(e.exceptions) == 1:
        e = e.exceptions[0]
    return e


#: How long one tool call may take over MCP. Longer than any tool's own limit
#: (sleep_for allows up to 300 s), so the tool's limit fires first and its message
#: reaches the model, instead of the bridge giving up on a call that was still fine.
CALL_TIMEOUT = 330


class MCPBridge:
    """Owns a persistent MCP session on a background event loop.

    **The whole session lives inside ONE task** (`_live`): it enters `stdio_client` and
    `ClientSession` with `async with`, signals ready, and then simply waits. Closing
    cancels that task, so the context managers are exited by the same task that
    entered them — the one rule anyio's structured concurrency enforces.

    The previous shape entered them by hand (`__aenter__`) in a connect coroutine and
    exited them by hand in a separate shutdown coroutine — a different task. anyio
    refused ("Attempted to exit cancel scope in a different task than it was entered
    in") on BOTH exits; close() swallowed that, and the server only died because the
    SDK happened to kill it before raising. Working by accident of the SDK's internal
    ordering is not a teardown anyone should rely on.

    A bridge that fails to connect cleans up before it raises: it used to leave its
    event-loop thread spinning forever.
    """

    def __init__(self, command: str, args: list[str], timeout: float = 30):
        self.tools = []
        self.session = None
        self.close_error: str | None = None     # a teardown problem, kept, never raised
        self.broken: str | None = None           # why the connection failed, if it has
        self._closed = False
        self._ready: concurrent.futures.Future = concurrent.futures.Future()
        self._finished: concurrent.futures.Future = concurrent.futures.Future()
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._spin, daemon=True,
                                        name=f"mcp-{Path(args[0]).stem if args else command}")
        self._thread.start()
        self._life = asyncio.run_coroutine_threadsafe(self._live(command, args), self._loop)
        try:
            self._ready.result(timeout=timeout)   # connected and tools discovered
        except BaseException:
            self.close()                          # never leave a half-built bridge running
            raise

    def _spin(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _submit(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    async def _live(self, command, args):
        """The session's entire lifetime, in one task. Ends when cancelled."""
        params = StdioServerParameters(command=command, args=args)
        try:
            async with stdio_client(params) as (read, write), \
                    ClientSession(read, write) as session:
                await session.initialize()
                self.tools = (await session.list_tools()).tools
                self.session = session
                self._ready.set_result(None)
                await asyncio.Event().wait()      # park here until close() cancels us
        except asyncio.CancelledError:
            pass                                  # the normal way out
        except BaseException as e:                # noqa: BLE001 - reported, not raised here
            e = _unwrap(e)
            if not self._ready.done():
                self._ready.set_exception(e)      # a failed connect surfaces in __init__
            else:
                self.close_error = f"{type(e).__name__}: {e}"
        finally:
            self.session = None
            self._finished.set_result(None)

    @property
    def alive(self) -> bool:
        """False once closed, or once a call failed at the connection level.

        A server that dies does not end the session on its own: the session task stays
        parked until close(). The first call after the death fails, and that failure
        is what marks the bridge broken, so an owner can replace it.
        """
        return self.session is not None and not self._closed and self.broken is None

    def call(self, name: str, args: dict) -> str:
        """Run one tool. Returns text, never raises: a timeout or a dead server is
        reported to the model as a string, like any other tool failure."""
        session = self.session
        if session is None:
            why = f" ({self.close_error})" if self.close_error else ""
            return f"error: the MCP server for {name} has stopped{why}"
        fut = self._submit(session.call_tool(name, args or {}))
        try:
            result = fut.result(timeout=CALL_TIMEOUT)
        except concurrent.futures.TimeoutError:
            fut.cancel()
            return f"error: {name} took longer than {CALL_TIMEOUT}s and was abandoned"
        except Exception as e:  # noqa: BLE001 - tools report, never raise
            # A tool's own failure comes back as a result, not an exception, so an
            # exception here means the connection itself failed.
            e = _unwrap(e)
            self.broken = f"{type(e).__name__}: {e}"
            return f"error: the MCP server for {name} failed: {self.broken}"
        # call_tool returns content blocks; pull text out of each
        return "\n".join(getattr(c, "text", str(c)) for c in result.content)

    def close(self) -> None:
        """Stop the session, the server subprocess and the loop thread. Never raises.

        Cancelling the lifetime task unwinds its `async with` blocks in that same task:
        ClientSession closes, then stdio_client closes the server's stdin, waits briefly
        for it to exit, and terminates it if it does not. Then the loop stops and its
        thread is joined, so nothing outlives the bridge. Safe to call twice.
        """
        if self._closed:
            return
        self._closed = True
        try:
            self._loop.call_soon_threadsafe(self._life.cancel)
            self._finished.result(timeout=15)
        except Exception as e:                     # noqa: BLE001 - teardown must not raise
            self.close_error = self.close_error or f"{type(e).__name__}: {e}"
        try:
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=5)
            if not self._thread.is_alive():
                self._loop.close()
        except Exception as e:                     # noqa: BLE001
            self.close_error = self.close_error or f"{type(e).__name__}: {e}"
        if self.close_error:
            # Not raised, but not silent either: a swallowed teardown error is how the
            # old close() hid that it only worked by accident.
            print(f"lag: MCP bridge teardown: {self.close_error}", file=sys.stderr)


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
        #
        # All or nothing: if a later server fails, the ones already started are closed
        # before the error propagates. Raising past them orphaned a live server and its
        # thread on every attempt — and the web console retries on every request.
        #
        # Each server gets the configured embed model and Ollama host as flags. Left to
        # its own defaults, the RAG server searched with nomic-embed-text whatever
        # `lag setup` had ingested with, and a store of another vector size broke.
        from .config import server_args

        flags = server_args(cfg)
        bridges = []
        try:
            for server in (servers or cls.DEFAULT_SERVERS):
                path = Path(server)
                if not path.is_absolute():
                    path = _REPO_ROOT / path
                bridges.append(MCPBridge(sys.executable, [str(path), *flags]))
        except BaseException:
            for b in bridges:
                b.close()
            raise
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

    def close(self) -> None:
        """Close every bridge. Safe to call twice, and never raises.

        Whoever builds a provider owns it: the CLI closes it when its command ends, the
        web server when it shuts down. Each bridge is two things that outlive a dropped
        reference — a server subprocess and an event-loop thread.
        """
        for b in self.bridges:
            with contextlib.suppress(Exception):   # one bad bridge must not strand the rest
                b.close()

    @property
    def alive(self) -> bool:
        """True while every server is still answering. A long-lived owner (the web
        console) checks this before reuse, so one crashed server does not leave every
        later request failing until a restart."""
        return all(getattr(b, "alive", True) for b in self.bridges)

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self.close()

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
