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
    raise FileNotFoundError("No config found. Run: cp config.example.yaml config.yaml")


def load_persona(name: str, cfg: dict) -> dict:
    """Load a persona definition from the configured personas directory."""
    base = Path(cfg.get("paths", {}).get("personas", "./personas"))
    p = base / f"{name}.yaml"
    if not p.exists():
        raise FileNotFoundError(f"persona not found: {p}")
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f)
