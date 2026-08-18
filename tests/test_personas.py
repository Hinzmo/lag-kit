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
        """Every declared tool must exist, except the known-missing web_search.

        web_search is a documented loose end (referenced by the research
        personas, never rebuilt from v1). This test allows it explicitly so the
        gap stays visible instead of silently widening to other typos.
        """
        from lag.tools import TOOLS

        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        declared = set(data.get("tools", []))
        unknown = declared - set(TOOLS) - {"web_search"}
        assert not unknown, f"{path.stem} declares unknown tools: {unknown}"


def test_load_persona_reads_from_configured_dir():
    from lag.config import load_persona

    cfg = {"paths": {"personas": str(PERSONA_DIR)}}
    assert load_persona("architect_agent", cfg)["name"] == "architect_agent"


def test_load_persona_raises_for_missing():
    from lag.config import load_persona

    with pytest.raises(FileNotFoundError):
        load_persona("no_such_agent", {"paths": {"personas": str(PERSONA_DIR)}})
