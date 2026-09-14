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

# Ollama reports what each tag can actually do, so capability is discovered, never a
# hardcoded list that rots the moment someone pulls a new model. Cached per process:
# a model's capabilities cannot change while it is installed, and show() is a round
# trip we would otherwise pay on every client construction.
_CAPS: dict[tuple[str, str], frozenset[str]] = {}


def capabilities(host: str, model: str) -> frozenset[str]:
    """What this tag supports — e.g. {"completion", "tools", "thinking"}.

    Returns an empty set if Ollama cannot be reached or does not know the tag. An
    empty set means "unknown", and callers must treat unknown as capable: refusing to
    pass tools because a status call failed would silently downgrade a working model.
    """
    key = (host, model)
    if key not in _CAPS:
        try:
            d = ollama.Client(host=host).show(model)
            _CAPS[key] = frozenset(getattr(d, "capabilities", None) or [])
        except Exception:  # noqa: BLE001 - capability probing must never break a run
            return frozenset()
    return _CAPS[key]


def supports(host: str, model: str, capability: str) -> bool:
    """True unless we positively know the model lacks the capability."""
    caps = capabilities(host, model)
    return not caps or capability in caps


class OllamaClient:
    def __init__(self, model: str, host: str = "http://localhost:11434",
                 system: str | None = None, num_ctx: int | None = None,
                 keep_alive: str | None = None, think: bool | None = None,
                 num_predict: int | None = None):
        """
        num_ctx matters more than it looks. Ollama does not use a model's trained
        context — it allocates 4096 by default regardless. ornith:latest is trained
        for 262144, so leaving this unset runs it at 1/64th of its window and
        silently drops the oldest tokens once the loop fills up. With always-on
        retrieval injecting chunks *and* tool schemas for nine tools, 4096 overflows
        quickly, and the symptom is a model that "forgets" the top of its own
        conversation rather than an error. Prefer from_config, which resolves the
        window per model.
        """
        self.model = model
        self.system = system
        self.num_ctx = num_ctx
        self.keep_alive = keep_alive
        # Personas have declared thinking_mode since the rebuild, but nothing ever
        # passed it to Ollama, so every agent reasoned at length whether its persona
        # asked for it or not — dev_agent (thinking_mode: false) was measured emitting
        # 2600+ thinking tokens before its first word. None = leave the model's default.
        # A hard ceiling on generation per turn. Without one a thinking model can
        # spiral: senior_dev_agent was measured emitting 4534 thinking tokens against
        # 16 of answer and never finishing a 15-minute run. num_predict bounds the
        # whole generation, thinking included, so a stage fails fast instead of hanging.
        self.num_predict = num_predict
        self.host = host
        self.client = ollama.Client(host=host)
        # A model without tool support cannot be fixed by a better prompt: the whole
        # ReAct loop is tool calls, and passing `tools` to a completion-only tag fails
        # the request outright. Dropping tools turns "the run dies" into "it answers
        # from what it knows", which is a usable model, just a smaller one.
        self.has_tools = supports(host, model, "tools")
        if think and not supports(host, model, "thinking"):
            think = None
        self.think = think

    @classmethod
    def from_config(cls, cfg: dict, model: str, system: str | None = None,
                    think: bool | None = None):
        """Build a client from the ollama: block, which was previously dead config.

        num_ctx is resolved per model, falling back to the global. Models differ
        enormously in what a token of window costs: measured here, ornith:latest holds
        131072 in ~10.1 GB while qwen3:8b needs ~9.9 GB for only 32768. Since exactly
        one chat model is resident at a time, each can spend the whole budget on its
        own terms rather than being pinned to the smaller model's ceiling.
        """
        o = cfg.get("ollama", {}) or {}
        per_model = o.get("num_ctx_by_model") or {}
        return cls(
            model=model,
            host=cfg.get("ollama_base_url", "http://localhost:11434"),
            system=system,
            num_ctx=per_model.get(model, o.get("num_ctx")),
            keep_alive=o.get("keep_alive"),
            think=think,
            num_predict=o.get("num_predict"),
        )

    def _chat_kwargs(self) -> dict:
        kw: dict = {}
        opts: dict = {}
        if self.num_ctx:
            opts["num_ctx"] = int(self.num_ctx)
        if self.num_predict:
            opts["num_predict"] = int(self.num_predict)
        if opts:
            kw["options"] = opts
        if self.keep_alive:
            kw["keep_alive"] = self.keep_alive
        if self.think is not None:
            kw["think"] = self.think
        return kw

    def _chat(self, **kw):
        """ollama.chat, retrying once without `think` if the model rejects it.

        Not every tag supports the parameter, and a model that does not should fall
        back to its default rather than fail the whole run — the same "degrade, do not
        raise" rule the tool layer follows.
        """
        try:
            return self.client.chat(**kw, **self._chat_kwargs())
        except Exception as e:  # noqa: BLE001
            if self.think is None or "think" not in str(e).lower():
                raise
            fallback = {k: v for k, v in self._chat_kwargs().items() if k != "think"}
            return self.client.chat(**kw, **fallback)

    def _turn(self, messages: list, schemas, on_event):
        """One assistant turn, streamed when someone is listening.

        Streaming is opt-in rather than the default because reconstructing tool_calls
        from chunks is strictly more fragile than reading them off a finished message,
        and the CLI has no use for deltas. When on_event is None this is the original
        single-shot call; the loop above cannot tell the difference either way, which
        is the same duck-typing rule tool_provider follows.
        """
        if on_event is None:
            return self._chat(model=self.model, messages=messages, tools=schemas).message

        content, thinking, tool_calls, final = [], [], [], None
        for chunk in self._chat(model=self.model, messages=messages, tools=schemas,
                                stream=True):
            final = chunk.message
            # Thinking tokens are surfaced separately: on a model like ornith they are
            # most of the wait, and a console showing nothing during them looks hung.
            if getattr(final, "thinking", None):
                thinking.append(final.thinking)
                on_event({"type": "thinking", "text": final.thinking})
            if final.content:
                content.append(final.content)
                on_event({"type": "token", "text": final.content})
            if final.tool_calls:
                tool_calls.extend(final.tool_calls)

        # Rebuild the turn the loop expects. ollama's chunks carry deltas, so the last
        # chunk alone would drop everything streamed before it.
        final.content = "".join(content)
        if thinking:
            final.thinking = "".join(thinking)
        final.tool_calls = tool_calls or None
        return final

    def run(self, prompt: str, tool_provider=None, max_iters: int = 8,
            verbose: bool = True, on_event=None) -> str:
        """Reason-act-observe until the model answers without calling a tool.

        on_event, when given, receives {"type": ...} dicts as they happen — token and
        thinking deltas, tool calls and their results. It is a side channel only: the
        return value is identical with or without it, so callers that do not care are
        unaffected.
        """
        messages = []
        if self.system:
            messages.append({"role": "system", "content": self.system})
        messages.append({"role": "user", "content": prompt})

        schemas = tool_provider.schemas() if tool_provider else None
        if schemas and not self.has_tools:
            # Say so once, loudly enough to explain a thin answer, then carry on.
            note = (f"{self.model} does not support tool calling — running without "
                    f"tools (no file access, shell, retrieval or memory this turn)")
            if verbose and on_event is None:
                print(f"  ! {note}")
            if on_event:
                on_event({"type": "no_tools", "model": self.model, "message": note})
            schemas = None

        malformed = 0
        for _ in range(max_iters):
            try:
                msg = self._turn(messages, schemas, on_event)
            except Exception as e:  # noqa: BLE001
                # A turn can fail on something the model did, not something we did:
                # small models emit malformed JSON for tool arguments, and ollama
                # raises while parsing it. Killing the whole run for that wastes every
                # step taken so far, and the model can usually recover if simply told.
                # Same rule as the tool layer: hand the error back as text.
                malformed += 1
                detail = f"{type(e).__name__}: {e}"
                if on_event:
                    on_event({"type": "turn_error", "message": detail,
                              "attempt": malformed})
                if malformed >= 3:
                    return f"(stopped: {malformed} malformed turns; last was {detail})"
                messages.append({
                    "role": "user",
                    "content": ("Your last reply could not be parsed: " + detail +
                                ". Reply again. If you are calling a tool, emit valid "
                                "JSON arguments with only the documented fields."),
                })
                continue
            messages.append(msg)  # the assistant turn, possibly carrying tool_calls

            if not msg.tool_calls:
                return msg.content or ""

            for call in msg.tool_calls:
                name = call.function.name
                args = call.function.arguments or {}
                # Report a tool call ONCE. A listener is already a reporting channel,
                # so printing as well double-logs every call to anyone using both.
                if verbose and on_event is None:
                    print(f"  \u21b3 {name}({args})")
                if on_event:
                    on_event({"type": "tool", "name": name, "args": args})
                result = tool_provider.call(name, args)
                if on_event:
                    on_event({"type": "tool_result", "name": name,
                              "content": str(result)[:2000]})
                messages.append({"role": "tool", "tool_name": name, "content": str(result)})

        return "(stopped: hit max_iters without a final answer)"
