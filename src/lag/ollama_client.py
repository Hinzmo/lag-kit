"""OllamaClient: drives a reason-act-observe loop against a local model.

The loop is deliberately agnostic about where tools come from. It talks to a
`tool_provider` that exposes two methods:
    schemas() -> list  (whatever ollama.chat accepts as `tools`)
    call(name, args) -> str
LocalToolProvider (tools.py) and MCPToolProvider (mcp_client.py) both satisfy it,
so swapping direct dispatch for a real MCP server is a one-line change at the call
site, not a rewrite of the loop.
"""
from __future__ import annotations
import ollama


class OllamaClient:
    def __init__(self, model: str, host: str = "http://localhost:11434",
                 system: str | None = None):
        self.model = model
        self.system = system
        self.client = ollama.Client(host=host)

    def run(self, prompt: str, tool_provider=None, max_iters: int = 8,
            verbose: bool = True) -> str:
        messages = []
        if self.system:
            messages.append({"role": "system", "content": self.system})
        messages.append({"role": "user", "content": prompt})

        schemas = tool_provider.schemas() if tool_provider else None

        for _ in range(max_iters):
            resp = self.client.chat(model=self.model, messages=messages, tools=schemas)
            msg = resp.message
            messages.append(msg)  # the assistant turn, possibly carrying tool_calls

            if not msg.tool_calls:
                return msg.content or ""

            for call in msg.tool_calls:
                name = call.function.name
                args = call.function.arguments or {}
                if verbose:
                    print(f"  \u21b3 {name}({args})")
                result = tool_provider.call(name, args)
                messages.append({"role": "tool", "tool_name": name, "content": str(result)})

        return "(stopped: hit max_iters without a final answer)"
