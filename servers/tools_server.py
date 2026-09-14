"""MCP server exposing local-agent-kit's tools over stdio.

Run standalone for inspection:
    python servers/tools_server.py
    # or poke it with the MCP Inspector:
    # npx @modelcontextprotocol/inspector python servers/tools_server.py

The harness normally spawns this itself via mcp_client.MCPBridge. Note we register
the *same* functions from lag.tools — no duplicated logic, the server is just a
protocol wrapper around them.
"""
try:
    # mcp >= 2.0 renamed the high-level interface: mcp.server.fastmcp.FastMCP
    # became mcp.server.mcpserver.MCPServer. The .tool() decorator and the
    # stdio-by-default .run() are unchanged, so both satisfy the usage below.
    from mcp.server.mcpserver import MCPServer as _Server
except ImportError:  # mcp 1.x
    from mcp.server.fastmcp import FastMCP as _Server

# One implementation, two consumers: these are the same functions the in-process
# provider dispatches to, wrapped unchanged. Importing TOOLS keeps the two lists
# from drifting — a tool added to tools.py is exposed here automatically.
from lag.tools import TOOLS

mcp = _Server("lag-tools")

for _fn in TOOLS.values():
    mcp.tool()(_fn)  # FastMCP builds the schema from each function's hints + docstring

if __name__ == "__main__":
    mcp.run()  # stdio transport by default
