"""`lag setup`: install, pull and map, without touching a real Ollama.

The rule worth pinning is precedence: a flag wins, then config.yaml, then the
defaults. Re-running setup with no flags must never overwrite a mapping you chose,
and it must pull what config.yaml names rather than the defaults.
"""
from __future__ import annotations

from types import SimpleNamespace as NS

import pytest
from typer.testing import CliRunner

from lag import setup as s
from lag.cli import app

EXAMPLE = '''ollama_base_url: "http://127.0.0.1:11434"
models:
  orchestrator: "qwen3:8b"      # plans
  worker: "qwen3:8b"            # shared
  coder: "qwen3:8b"
  fast: "qwen3:8b"
  embed: "nomic-embed-text"     # embeddings

pipeline:
  default_agents: "architect_agent,dev_agent"
  # worker: "not a role line, outside models:"
'''


class FakeClient:
    def __init__(self, have=(), up=True, bad=()):
        self.have, self.up, self.bad, self.pulled = set(have), up, set(bad), []

    def list(self):
        if not self.up:
            raise ConnectionError("down")
        return NS(models=[NS(model=m) for m in sorted(self.have)])

    def pull(self, tag, stream=True):
        if tag in self.bad:
            raise RuntimeError("file does not exist")
        self.pulled.append(tag)
        self.have.add(s.norm(tag))
        yield NS(status="pulling", total=100, completed=50)
        yield NS(status="success", total=None, completed=None)


# ── pure pieces ─────────────────────────────────────────────────────────────

def test_norm_treats_bare_names_as_latest():
    assert s.norm("nomic-embed-text") == "nomic-embed-text:latest"
    assert s.norm("qwen3:8b") == "qwen3:8b"


def test_plan_roles_flag_beats_config_beats_defaults():
    assert s.plan_roles({}, None, None, None) == {
        "orchestrator": "qwen3.5:9b", "worker": "qwen3.5:9b", "coder": "qwen3.5:9b",
        "fast": "qwen3.5:9b", "embed": "nomic-embed-text"}
    mine = {"orchestrator": "a:1", "worker": "b:1", "coder": "c:1", "fast": "d:1", "embed": "e"}
    assert s.plan_roles(mine, None, None, None) == mine
    r = s.plan_roles(mine, None, "z:2", None)
    assert r["orchestrator"] == "a:1" and r["embed"] == "e"
    assert r["worker"] == r["coder"] == r["fast"] == "z:2"


def test_write_models_changes_only_role_lines_inside_models():
    roles = s.plan_roles({}, "big:9b", "small:4b", None)
    out = s.write_models(EXAMPLE, roles)
    assert '  orchestrator: "big:9b"      # plans\n' in out
    assert '  worker: "small:4b"            # shared\n' in out
    assert '  coder: "small:4b"\n' in out
    assert '  embed: "nomic-embed-text"     # embeddings\n' in out
    assert '  # worker: "not a role line, outside models:"' in out
    assert out.count("\n") == EXAMPLE.count("\n")


def test_write_models_is_stable():
    roles = s.plan_roles({}, None, None, None)
    once = s.write_models(EXAMPLE, roles)
    assert s.write_models(once, roles) == once


def test_install_command_per_platform():
    has = lambda name: f"/usr/bin/{name}"  # noqa: E731
    none = lambda name: None  # noqa: E731
    assert s.install_command("Darwin", has) == ["brew", "install", "ollama"]
    assert s.install_command("Darwin", none) is None
    assert s.install_command("Linux", has)[-1].endswith("install.sh | sh")
    assert s.install_command("Windows", has) is None


def test_start_server_waits_until_it_answers():
    client = FakeClient(up=False)
    started = []

    def popen(cmd, **kw):
        started.append(cmd)
        client.up = True

    assert s.start_server("/bin/ollama", client, popen=popen, sleep=lambda _: None)
    assert started == [["/bin/ollama", "serve"]]


def test_pull_echoes_progress_without_repeats():
    lines = []
    s.pull(FakeClient(), "m:1", lines.append)
    assert lines == ["    m:1: pulling 50%", "    m:1: success"]


# ── the command ─────────────────────────────────────────────────────────────

@pytest.fixture
def repo(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.example.yaml").write_text(EXAMPLE)
    monkeypatch.setattr(s, "find_ollama", lambda: "/bin/ollama")
    monkeypatch.setattr("lag.ollama_client.capabilities", lambda h, m: frozenset({"tools"}))
    client = FakeClient(have={"qwen3:8b"})
    import ollama
    monkeypatch.setattr(ollama, "Client", lambda host=None: client)
    return tmp_path, client


def test_first_run_pulls_defaults_and_writes_config(repo):
    path, client = repo
    r = CliRunner().invoke(app, ["setup", "--no-ingest"])
    assert r.exit_code == 0, r.output
    assert client.pulled == ["qwen3.5:9b", "nomic-embed-text"]
    assert 'orchestrator: "qwen3.5:9b"' in (path / "config.yaml").read_text()


def test_rerun_keeps_your_mapping_and_pulls_it(repo):
    path, client = repo
    (path / "config.yaml").write_text(s.write_models(EXAMPLE, s.plan_roles({}, "mine:7b", "mine-chat:4b", None)))
    before = (path / "config.yaml").read_text()
    r = CliRunner().invoke(app, ["setup", "--no-ingest"])
    assert r.exit_code == 0, r.output
    assert {"mine:7b", "mine-chat:4b"} <= set(client.pulled) and "qwen3.5:9b" not in client.pulled
    assert (path / "config.yaml").read_text() == before
    assert "unchanged" in r.output


def test_a_bad_tag_fails_without_writing_config(repo):
    path, client = repo
    client.bad.add("typo:1b")
    r = CliRunner().invoke(app, ["setup", "--no-ingest", "--model", "typo:1b"])
    assert r.exit_code == 1
    assert "Could not pull typo:1b" in r.output
    assert not (path / "config.yaml").exists()


def test_missing_ollama_asks_before_installing(repo, monkeypatch):
    monkeypatch.setattr(s, "find_ollama", lambda: None)
    monkeypatch.setattr(s, "install_command", lambda: ["brew", "install", "ollama"])
    ran = []
    monkeypatch.setattr("lag.cli.subprocess.run", lambda cmd: ran.append(cmd))
    r = CliRunner().invoke(app, ["setup", "--no-ingest"], input="n\n")
    assert r.exit_code == 1 and ran == []
    assert "ollama.com/download" in r.output


def test_warns_when_a_chat_model_has_no_tools(repo, monkeypatch):
    monkeypatch.setattr("lag.ollama_client.capabilities", lambda h, m: frozenset({"completion"}))
    r = CliRunner().invoke(app, ["setup", "--no-ingest"])
    assert r.exit_code == 0
    assert "does not support tools" in r.output


def test_rival_servers_spots_two_ollamas_that_disagree():
    lists = {"http://127.0.0.1:11434": {"a:1", "b:1"}, "http://[::1]:11434": {"a:1"}}
    make = lambda h: FakeClient(have=lists[h])  # noqa: E731
    assert s.rival_servers("http://127.0.0.1:11434", make)
    lists["http://[::1]:11434"] = {"a:1", "b:1"}
    assert not s.rival_servers("http://localhost:11434", make)


def test_rival_servers_quiet_for_one_server_or_a_remote_host():
    def make(h):
        if "::1" in h:
            return FakeClient(up=False)
        return FakeClient(have={"a:1"})
    assert not s.rival_servers("http://127.0.0.1:11434", make)
    assert not s.rival_servers("http://gpu-box:11434", lambda h: 1 / 0)
