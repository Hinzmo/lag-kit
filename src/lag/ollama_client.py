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
                 system: str | None = None, num_ctx: int | None = None,
                 keep_alive: str | None = None):
        """
        num_ctx matters more than it looks. Ollama does not use a model's trained
        context — it allocates 4096 by default regardless. ministral-3:8b is
        trained for 262144, so leaving this unset runs it at 1/64th of its window
        and silently drops the oldest tokens once the loop fills up. With
        always-on retrieval injecting chunks *and* tool schemas for eight tools,
        4096 overflows quickly, and the symptom is a model that "forgets" the
        top of its own conversation rather than an error.
        """
        self.model = model
        self.system = system
        self.num_ctx = num_ctx
        self.keep_alive = keep_alive
        self.client = ollama.Client(host=host)

    @classmethod
    def from_config(cls, cfg: dict, model: str, system: str | None = None):
        """Build a client from the ollama: block, which was previously dead config."""
        o = cfg.get("ollama", {}) or {}
        return cls(
            model=model,
            host=cfg.get("ollama_base_url", "http://localhost:11434"),
            system=system,
            num_ctx=o.get("num_ctx"),
            keep_alive=o.get("keep_alive"),
        )

    def _chat_kwargs(self) -> dict:
        kw: dict = {}
        if self.num_ctx:
            kw["options"] = {"num_ctx": int(self.num_ctx)}
        if self.keep_alive:
            kw["keep_alive"] = self.keep_alive
        return kw

    def run(self, prompt: str, tool_provider=None, max_iters: int = 8,
            verbose: bool = True) -> str:
        messages = []
        if self.system:
            messages.append({"role": "system", "content": self.system})
        messages.append({"role": "user", "content": prompt})

        schemas = tool_provider.schemas() if tool_provider else None

        for _ in range(max_iters):
            resp = self.client.chat(model=self.model, messages=messages, tools=schemas,
                                    **self._chat_kwargs())
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
