"""Configuration and persona loading for local-agent-kit."""
from __future__ import annotations
from pathlib import Path
import yaml

_SEARCH = ["config.yaml", "config.example.yaml"]


def load_config(path: str | None = None) -> dict:
    """Load the YAML config.

    Falls back to config.example.yaml if config.yaml hasn't been created yet,
    so the kit runs out of the box before you've customised it per machine.
    """
    for candidate in ([path] if path else _SEARCH):
        if candidate and Path(candidate).exists():
            with open(candidate, encoding="utf-8") as f:
                return yaml.safe_load(f)
    raise FileNotFoundError("No config found. Run lag setup from the repo root.")


def embed_settings(cfg: dict) -> dict:
    """The embed model and Ollama host from config, as keyword arguments for rag.

    Only keys the config sets are returned, so rag's own defaults fill the rest. Every
    process that embeds — `lag setup`'s ingest, the RAG server, `remember` — must use
    the same model: a store built at one vector size cannot be searched at another.
    """
    out = {}
    model = (cfg.get("models") or {}).get("embed")
    if model:
        out["model"] = model
    if cfg.get("ollama_base_url"):
        out["host"] = cfg["ollama_base_url"]
    return out


def server_args(cfg: dict) -> list[str]:
    """embed_settings as command-line flags for the MCP servers."""
    s = embed_settings(cfg)
    return ([f"--embed-model={s['model']}"] if "model" in s else []) + \
           ([f"--ollama-host={s['host']}"] if "host" in s else [])


def resolve_model(spec: str, cfg: dict) -> str:
    """Resolve a persona's `model:` field to a concrete Ollama tag.

    Personas name a *role* — `coder`, `worker`, `orchestrator` — which resolves
    through `models:` in config.yaml. Model names then live in exactly one place,
    so swapping in a new model is a one-line config change instead of an edit
    across nine persona files. It also lets each machine run a different model
    set from the same personas: the Mac can point `coder` somewhere the CPU-only
    box never could.

    A literal tag still works and passes through untouched, so older personas
    keep loading.
    """
    models = cfg.get("models", {}) or {}
    if spec in models:
        return models[spec]
    if ":" not in str(spec):
        # No colon and not a known role: almost certainly a mistyped role, which
        # would otherwise surface as a confusing 404 from Ollama.
        raise ValueError(
            f"persona model {spec!r} is not a role in config.yaml "
            f"(have: {', '.join(sorted(models)) or 'none'}); "
            f"use a role name, or a literal tag like 'qwen3.5:4b'"
        )
    return spec


def load_persona(name: str, cfg: dict) -> dict:
    """Load a persona definition, resolving its model role against the config."""
    base = Path(cfg.get("paths", {}).get("personas", "./personas"))
    p = base / f"{name}.yaml"
    if not p.exists():
        raise FileNotFoundError(f"persona not found: {p}")
    with open(p, encoding="utf-8") as f:
        persona = yaml.safe_load(f)
    if persona.get("model"):
        persona["model"] = resolve_model(persona["model"], cfg)
    return persona
