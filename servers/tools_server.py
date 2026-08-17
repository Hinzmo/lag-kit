"""MCP server exposing local-agent-kit's tools over stdio.

Run standalone for inspection:
    python servers/tools_server.py
    # or poke it with the MCP Inspector:
    # npx @modelcontextprotocol/inspector python servers/tools_server.py

The harness normally spawns this itself via mcp_client.MCPBridge. Note we register
the *same* functions from lag.tools — no duplicated logic, the server is just a
protocol wrapper around them.
"""
from mcp.server.fastmcp import FastMCP

from lag.tools import (
    read_file,
    write_file,
    list_dir,
    run_shell,
    system_info,
    remember,
    recall,
)

mcp = FastMCP("lag-tools")

for _fn in (read_file, write_file, list_dir, run_shell, system_info, remember, recall):
    mcp.tool()(_fn)  # FastMCP builds the schema from each function's hints + docstring

if __name__ == "__main__":
    mcp.run()  # stdio transport by default
