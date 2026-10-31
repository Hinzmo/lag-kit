"""Wrappers around a tool_provider.

The harness has one tool interface — `schemas()` and `call(name, args)` — and several
things that need to sit in front of it: an allow-list, a deny-list, keeping the goal
tools in this process, and confining the dangerous ones to a container. Each is a
wrapper rather than a replacement, so the agent loop never learns which are present and
they compose in any order.

They live together here because they are one family with one contract. They grew up
scattered — three in agent.py and one in sandbox.py — which meant web.py imported a
module about *agent loops* purely to filter a tool out of a pipeline run.

**Every filter must read both schema shapes.** LocalToolProvider hands back the tool
functions themselves; MCPToolProvider hands back Ollama-shaped dicts. A filter that
understands only one looks enforced in tests and silently passes everything over the
other transport — which, for the sandbox, would be the difference between confined and
not.
"""
from __future__ import annotations


def schema_name(schema) -> str:
    """The tool name out of either provider's schema shape."""
    if callable(schema):
        return getattr(schema, "__name__", "")
    if isinstance(schema, dict):
        return (schema.get("function") or {}).get("name") or schema.get("name") or ""
    return ""


class ProviderWrapper:
    """Pass-through base. Subclasses override only what they change."""

    def __init__(self, inner):
        self.inner = inner

    def schemas(self):
        return self.inner.schemas()

    def call(self, name: str, args: dict) -> str:
        return self.inner.call(name, args)


class _ToolFilter(ProviderWrapper):
    """Shared machinery for the allow-list and deny-list.

    A refused call returns a string the model can read and adapt to, never an
    exception — the same rule the tool layer itself follows. An agent told a tool is
    unavailable picks another approach; one that gets a traceback stops.
    """

    def _permitted(self, name: str) -> bool:
        raise NotImplementedError

    def _refusal(self, name: str) -> str:
        raise NotImplementedError

    def schemas(self):
        return [s for s in self.inner.schemas() if self._permitted(schema_name(s))]

    def call(self, name: str, args: dict) -> str:
        if not self._permitted(name):
            return self._refusal(name)
        return self.inner.call(name, args)


class ToolSubset(_ToolFilter):
    """Narrow a provider to an allow-list. An EMPTY list means every tool.

    That default matters: it is what lets an agent definition leave `tools:` unset and
    get everything, so narrowing is always a deliberate act rather than something
    inherited from a template.
    """

    def __init__(self, inner, allow: list[str]):
        super().__init__(inner)
        self.allow = set(allow)

    def _permitted(self, name: str) -> bool:
        return not self.allow or name in self.allow

    def _refusal(self, name: str) -> str:
        return (f"blocked: {name} is not in this agent's allowed tools "
                f"({', '.join(sorted(self.allow))})")


class ToolsExcept(_ToolFilter):
    """Withhold named tools. The inverse of ToolSubset, and the reason it exists:
    turning retrieval off should remove one tool, not require listing the other
    fifteen."""

    def __init__(self, inner, deny: list[str]):
        super().__init__(inner)
        self.deny = set(deny)

    def _permitted(self, name: str) -> bool:
        return name not in self.deny

    def _refusal(self, name: str) -> str:
        return f"blocked: {name} is switched off for this run"
