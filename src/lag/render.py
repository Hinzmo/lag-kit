"""Turn raw model and tool output into something a person reads.

Nothing here changes what the model is *given* — only what gets shown. It exists
because three different things were putting raw JSON in front of the user:

  - tool calls, rendered as their argument dict (`{"path": "src/lag"}`)
  - tools that genuinely return JSON (`system_info`)
  - models that fumble a tool call and emit it as text instead, so the JSON lands in
    the answer

The rule is that parsing happens ONCE, upstream, before any consumer sees it — the
chat pane, the activity window and the CLI all render the same cleaned text. Putting
it in the view layer instead would mean three implementations that drift, and the
activity window would keep showing raw JSON precisely because it is the "debug" view.

Everything is defensive: a cleaner that throws, or that mangles a legitimate answer
containing a code block, is worse than one that occasionally leaves something through.
"""
from __future__ import annotations

import json
import re

# <think>…</think> should never reach content — ollama surfaces reasoning on its own
# channel — but models emit it inline anyway, especially when a tool call confuses them.
_THINK = re.compile(r"<think>.*?</think>", re.S | re.I)
_OPEN_THINK = re.compile(r"<think>.*\Z", re.S | re.I)
# A fenced block that is *only* a tool call, e.g. ```json {"name": "read_file", ...}```
_TOOLCALL_FENCE = re.compile(
    r"```(?:json|tool_code|tool_call)?\s*(\{[^`]*?\})\s*```", re.S | re.I)
_TOOLCALL_KEYS = ({"name", "arguments"}, {"name", "parameters"},
                  {"tool", "arguments"}, {"function", "arguments"})


def _looks_like_tool_call(blob: str) -> bool:
    try:
        d = json.loads(blob)
    except ValueError:
        return False
    if not isinstance(d, dict):
        return False
    keys = set(d)
    return any(sig <= keys for sig in _TOOLCALL_KEYS)


def clean_text(text: str) -> str:
    """Strip reasoning leakage and tool-call JSON from an answer.

    Deliberately conservative: a fenced block is removed only when it parses as JSON
    *and* carries tool-call keys, so a legitimate ```json example in an answer about
    JSON survives. Real code blocks are never touched.
    """
    if not text:
        return ""
    out = _THINK.sub("", text)
    out = _OPEN_THINK.sub("", out)          # an unclosed block means the rest is thinking
    out = _TOOLCALL_FENCE.sub(
        lambda m: "" if _looks_like_tool_call(m.group(1)) else m.group(0), out)
    # A bare tool-call object on its own line, unfenced.
    out = "\n".join(
        line for line in out.splitlines()
        if not (line.strip().startswith("{") and _looks_like_tool_call(line.strip()))
    )
    return re.sub(r"\n{3,}", "\n\n", out).strip()


def format_call(name: str, args: dict | None) -> str:
    """A tool call as a readable signature: read_file(path="src/lag/agent.py").

    Long values are elided rather than wrapped, because the point of this line is to
    say *what happened*, not to reproduce the payload.
    """
    if not args:
        return f"{name}()"
    parts = []
    for k, v in args.items():
        if isinstance(v, str):
            shown = v if len(v) <= 60 else v[:57] + "…"
            parts.append(f'{k}="{shown}"')
        elif isinstance(v, (int, float, bool)) or v is None:
            parts.append(f"{k}={v}")
        else:
            blob = json.dumps(v, ensure_ascii=False)
            parts.append(f"{k}={blob if len(blob) <= 60 else blob[:57] + '…'}")
    return f"{name}({', '.join(parts)})"


def humanise(text: str, limit: int = 4000) -> str:
    """Render a tool result for a human.

    Tools return strings, but some of those strings are JSON — `system_info` hands back
    a raw object. Rendering it as `os: Darwin` beats showing braces and quotes, while
    anything that is not JSON passes through untouched.
    """
    if not text:
        return ""
    s = text.strip()
    if s[:1] in "{[":
        try:
            data = json.loads(s)
        except ValueError:
            return text[:limit]
        rendered = _render(data)
        if rendered is not None:
            return rendered[:limit]
    return text[:limit]


def _render(data, indent: int = 0) -> str | None:
    pad = "  " * indent
    if isinstance(data, dict):
        if not data:
            return f"{pad}(empty)"
        lines = []
        for k, v in data.items():
            if isinstance(v, (dict, list)):
                inner = _render(v, indent + 1)
                lines.append(f"{pad}{k}:\n{inner}" if inner else f"{pad}{k}:")
            else:
                lines.append(f"{pad}{k}: {v}")
        return "\n".join(lines)
    if isinstance(data, list):
        if not data:
            return f"{pad}(none)"
        if all(not isinstance(x, (dict, list)) for x in data):
            return "\n".join(f"{pad}- {x}" for x in data)
        return "\n".join(_render(x, indent) or "" for x in data)
    return None


class StreamCleaner:
    """Strip `<think>` spans from a token stream without waiting for the end.

    Deltas arrive a few characters at a time, so a tag can straddle two of them. Text
    that might be the start of a tag is held back until it is resolved rather than
    emitted and retracted — a console cannot un-print a token.
    """

    _MAX_HOLD = 16          # longer than "</think>", so a real tag always resolves

    def __init__(self) -> None:
        self.in_think = False
        self._buf = ""

    def feed(self, delta: str) -> str:
        self._buf += delta
        out = []
        while self._buf:
            if self.in_think:
                end = self._buf.lower().find("</think>")
                if end == -1:
                    # Keep only enough to recognise a closing tag split across deltas.
                    self._buf = self._buf[-self._MAX_HOLD:]
                    break
                self._buf = self._buf[end + len("</think>"):]
                self.in_think = False
                continue
            start = self._buf.lower().find("<think>")
            if start != -1:
                out.append(self._buf[:start])
                self._buf = self._buf[start + len("<think>"):]
                self.in_think = True
                continue
            # No tag: emit everything except a possible partial one at the tail.
            keep = 0
            for n in range(1, min(len(self._buf), self._MAX_HOLD) + 1):
                if "<think>".startswith(self._buf[-n:].lower()):
                    keep = n
            if keep:
                out.append(self._buf[:-keep])
                self._buf = self._buf[-keep:]
            else:
                out.append(self._buf)
                self._buf = ""
            break
        return "".join(out)

    def flush(self) -> str:
        """Whatever is still held back, once the stream ends."""
        tail = "" if self.in_think else self._buf
        self._buf = ""
        return tail
