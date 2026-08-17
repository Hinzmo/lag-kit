"""`lag` command-line interface."""
from __future__ import annotations
import typer

from .config import load_config
from .ollama_client import OllamaClient
from .tools import LocalToolProvider

app = typer.Typer(help="local-agent-kit \u2014 offline multi-agent harness over Ollama.")


@app.command()
def run(
    prompt: str,
    model: str = typer.Option(None, help="Override the configured model."),
    mcp: bool = typer.Option(False, "--mcp", help="Route tools through the MCP server instead of direct dispatch."),
):
    """Run a single agent on a prompt."""
    cfg = load_config()
    model = model or cfg["models"]["worker"]

    if mcp:
        from .mcp_client import MCPToolProvider
        provider = MCPToolProvider.from_config(cfg)
    else:
        provider = LocalToolProvider()

    client = OllamaClient(model=model, host=cfg["ollama_base_url"])
    print(client.run(prompt, tool_provider=provider))


@app.command()
def swarm(
    task: str,
    mode: str = typer.Option(None, help="pipeline | parallel | crew"),
    agents: str = typer.Option("architect_agent,dev_agent,senior_dev_agent", help="Comma-separated persona names (first is coordinator in crew mode)."),
):
    """Run a multi-agent swarm on a task."""
    from .swarm import Swarm
    cfg = load_config()
    mode = mode or cfg["swarm"]["default_mode"]
    names = [a.strip() for a in agents.split(",")]
    print(Swarm.from_personas(names, cfg).run(task, mode=mode))


if __name__ == "__main__":
    app()
