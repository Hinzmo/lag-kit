"""Persona tests.

The nine persona YAMLs were the one piece feared lost in the wipe, so these
tests pin the contract `config.load_persona` and `swarm.Agent` rely on. A
persona that loads but lacks `model` or `system_prompt` fails much later, deep
inside a swarm run, where the traceback is far less obvious.
"""
from __future__ import annotations
from pathlib import Path

import pytest
import yaml

PERSONA_DIR = Path(__file__).resolve().parents[1] / "personas"
PERSONA_FILES = sorted(PERSONA_DIR.glob("*.yaml"))

# The crew named in MORNING_BRIEF.md; cli.py defaults to the first three.
NAMED_CREW = {"architect_agent", "dev_agent", "senior_dev_agent", "design_agent"}


def test_all_nine_personas_present():
    assert len(PERSONA_FILES) == 9


def test_named_crew_present():
    stems = {p.stem for p in PERSONA_FILES}
    assert NAMED_CREW <= stems


@pytest.mark.parametrize("path", PERSONA_FILES, ids=lambda p: p.stem)
class TestEachPersona:
    def test_has_required_keys(self, path):
        # swarm.Agent.__init__ reads name/model/system_prompt directly.
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert {"name", "model", "system_prompt"} <= set(data)

    def test_name_matches_filename(self, path):
        # load_persona resolves personas/<name>.yaml, so a mismatch means the
        # persona can never be loaded by its own name.
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert data["name"] == path.stem

    def test_system_prompt_is_nonempty_string(self, path):
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert isinstance(data["system_prompt"], str)
        assert data["system_prompt"].strip()

    def test_declared_tools_are_known(self, path):
        """Every declared tool must actually exist.

        This previously carried an exemption for web_search, which the research
        personas declared but nothing implemented. It is implemented now, so the
        exemption is gone and no persona may reference a tool that isn't real.
        """
        from lag.tools import TOOLS

        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        unknown = set(data.get("tools", [])) - set(TOOLS)
        assert not unknown, f"{path.stem} declares unknown tools: {unknown}"


def test_research_personas_are_no_longer_degraded():
    """research_agent and dev_research_agent were the two agents web_search broke."""
    from lag.tools import TOOLS

    for name in ("research_agent", "dev_research_agent"):
        data = yaml.safe_load((PERSONA_DIR / f"{name}.yaml").read_text(encoding="utf-8"))
        assert "web_search" in data["tools"]
        assert "web_search" in TOOLS


def test_load_persona_reads_from_configured_dir():
    from lag.config import load_persona

    cfg = {"paths": {"personas": str(PERSONA_DIR)}}
    assert load_persona("architect_agent", cfg)["name"] == "architect_agent"


def test_load_persona_raises_for_missing():
    from lag.config import load_persona

    with pytest.raises(FileNotFoundError):
        load_persona("no_such_agent", {"paths": {"personas": str(PERSONA_DIR)}})
