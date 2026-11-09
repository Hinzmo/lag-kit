"""Phone access: pairing, device keys, the phone port, and sandboxed phone chats.

The phone reaches lag serve through `tailscale serve`, which forwards from 127.0.0.1,
so by address a phone looks exactly like the Mac's own browser. These tests pin the
things that keep that safe: a key on every phone-port request, a key stored only as a
hash, a code that works once, revoking from the Mac only, and phone chats that never
get the Mac's own shell.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from lag import phone, web  # noqa: E402

PORT = 8766


@pytest.fixture(autouse=True)
def phone_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(phone, "PHONE_DIR", tmp_path / "phone")
    monkeypatch.setattr(web, "PHONE_PORT", PORT)
    phone._last_touch.clear()
    return tmp_path / "phone"


def _paired(name="Pixel 8", login=""):
    code, _ = phone.new_code()
    key, device = phone.redeem(code, name, login)
    return key, device


console = TestClient(web.app)                                   # port 80
handset = TestClient(web.app, base_url=f"http://testserver:{PORT}")


def _auth(key):
    return {"Authorization": f"Bearer {key}"}


# ── pairing codes and device keys ─────────────────────────────────────────────

class TestPairing:
    def test_a_code_trades_for_a_key_once(self):
        code, _ = phone.new_code()
        key, device = phone.redeem(code, "Pixel 8")
        assert device["name"] == "Pixel 8"
        assert phone.lookup(key)["name"] == "Pixel 8"
        assert "no pairing code is active" in phone.redeem(code, "again")

    def test_only_the_hash_of_the_key_is_kept(self, phone_dir):
        key, _ = _paired()
        stored = (phone_dir / "devices.json").read_text()
        assert key not in stored
        assert "key_sha256" in stored
        assert "key_sha256" not in json.dumps(phone.devices())

    def test_the_files_are_private(self, phone_dir):
        _paired()
        assert (phone_dir / "devices.json").stat().st_mode & 0o777 == 0o600

    def test_a_code_typed_by_hand_still_works(self):
        code, _ = phone.new_code()
        typed = phone.show_code(code).lower().replace("0", "o").replace("1", "l")
        assert not isinstance(phone.redeem(f"  {typed} ", "p"), str)

    def test_wrong_guesses_use_up_the_code(self):
        code, _ = phone.new_code()
        for left in range(phone.CODE_TRIES - 1, 0, -1):
            assert f"{left} tries left" in phone.redeem("ZZZZZZZZ", "p")
        assert "no tries left" in phone.redeem("ZZZZZZZZ", "p")
        assert "no pairing code is active" in phone.redeem(code, "p")   # even the right one

    def test_a_code_expires(self, monkeypatch):
        code, expires = phone.new_code(minutes=5)
        monkeypatch.setattr(phone, "_now", lambda: expires + 1)
        assert "expired" in phone.redeem(code, "p")

    def test_a_new_code_replaces_the_old_one(self):
        old, _ = phone.new_code()
        phone.new_code()
        assert isinstance(phone.redeem(old, "p"), str)

    def test_two_phones_with_one_name_stay_apart(self):
        _paired("Pixel 8")
        _, second = _paired("Pixel 8")
        assert second["name"] == "Pixel 8 (2)"

    def test_revoking_stops_the_key(self):
        key, _ = _paired()
        assert phone.revoke("Pixel 8")
        assert phone.lookup(key) is None
        assert not phone.revoke("Pixel 8")

    def test_the_refused_log_never_holds_a_key(self, phone_dir):
        phone.log_refused("/api/state", "unknown key", "a@example.com")
        line = json.loads((phone_dir / "refused.log").read_text())
        assert set(line) == {"at", "path", "reason", "tailscale_login"}


# ── the phone port ────────────────────────────────────────────────────────────

class TestPhonePort:
    def test_no_key_no_entry(self, phone_dir):
        assert handset.get("/api/devices").status_code == 401
        assert handset.get("/api/devices", headers=_auth("made-up")).status_code == 401
        reasons = [json.loads(x)["reason"]
                   for x in (phone_dir / "refused.log").read_text().splitlines()]
        assert reasons == ["no key", "unknown key"]

    def test_pair_then_use_the_key(self):
        code, _ = phone.new_code()
        r = handset.post("/api/pair", json={"code": code, "name": "Pixel 8"})
        assert r.status_code == 200
        key = r.json()["key"]
        me = handset.get("/api/devices", headers=_auth(key))
        assert me.status_code == 200 and me.json()["you"] == "Pixel 8"

    def test_a_bad_code_is_refused_and_logged(self, phone_dir):
        phone.new_code()
        r = handset.post("/api/pair", json={"code": "ZZZZZZZZ", "name": "x"})
        assert r.status_code == 403 and "tries left" in r.json()["error"]
        assert (phone_dir / "refused.log").exists()

    def test_pairing_is_only_on_the_phone_port(self):
        code, _ = phone.new_code()
        assert console.post("/api/pair", json={"code": code}).status_code == 404

    def test_the_console_port_needs_no_key(self):
        assert console.get("/api/devices").status_code == 200

    def test_a_revoked_phone_is_refused_on_its_next_request(self):
        key, _ = _paired()
        assert handset.get("/api/devices", headers=_auth(key)).status_code == 200
        phone.revoke("Pixel 8")                     # the CLI, in another process
        assert handset.get("/api/devices", headers=_auth(key)).status_code == 401

    def test_phones_are_revoked_from_the_mac_only(self):
        key, _ = _paired()
        assert handset.delete("/api/devices/Pixel 8", headers=_auth(key)).status_code == 403
        assert console.delete("/api/devices/Pixel 8").json() == {"revoked": "Pixel 8"}
        assert console.delete("/api/devices/Pixel 8").status_code == 404

    def test_the_key_must_come_from_the_tailscale_user_who_paired_it(self):
        key, _ = _paired(login="me@example.com")
        mine = {**_auth(key), "Tailscale-User-Login": "me@example.com"}
        theirs = {**_auth(key), "Tailscale-User-Login": "someone@example.com"}
        assert handset.get("/api/devices", headers=mine).status_code == 200
        assert handset.get("/api/devices", headers=theirs).status_code == 403

    def test_the_tailnet_name_is_accepted_as_host(self):
        """Serve may pass the phone's address through as Host; the key is the lock."""
        key, _ = _paired()
        r = handset.get("/api/devices", headers={**_auth(key), "host": "mac.tail1.ts.net"})
        assert r.status_code == 200

    def test_a_cross_site_post_is_refused_even_with_a_key(self):
        key, _ = _paired()
        r = handset.post("/api/goals/pause",
                         headers={**_auth(key), "Origin": "http://evil.example"})
        assert r.status_code == 403


# ── phone chats run in the sandbox ────────────────────────────────────────────

class FakeBox:
    started = 0

    def __init__(self, workspace, read_root=None, network=False):
        self.workspace, self.network = workspace, network
        self.cli, self.container, self.running, self.stopped = "docker", "box", True, False

    def start(self):
        FakeBox.started += 1

    def stop(self):
        self.stopped = True

    def _docker(self, argv):
        return SimpleNamespace(returncode=0, stdout="true\n" if self.running else "false\n")


class TestPhoneChatsAreSandboxed:
    @pytest.fixture(autouse=True)
    def _fakes(self, monkeypatch):
        from lag import sandbox

        FakeBox.started = 0
        monkeypatch.setattr(sandbox, "Sandbox", FakeBox)
        monkeypatch.setattr(sandbox, "available", lambda: (True, ""))
        monkeypatch.setattr(web, "_phone_box", {})
        monkeypatch.setattr(web, "load_config", lambda: {"models": {"worker": "m:1"}})
        monkeypatch.setattr(web, "_provider", lambda cfg, mcp: SimpleNamespace(
            schemas=list, call=lambda n, a: "host shell!"))
        self.used = []

        class Client:
            @classmethod
            def from_config(cls, cfg, model):
                return cls()

            def run(inner, prompt, tool_provider=None, **kw):
                self.used.append(tool_provider)
                return "ok"

        monkeypatch.setattr(web, "OllamaClient", Client)

    def test_a_phone_chat_gets_the_sandboxed_shell(self):
        from lag.sandbox import SandboxedProvider

        key, _ = _paired()
        assert handset.post("/api/run", json={"prompt": "hi"}, headers=_auth(key)).json() \
            == {"answer": "ok", "model": "m:1"}
        assert isinstance(self.used[-1], SandboxedProvider)

    def test_a_chat_at_the_mac_is_unchanged(self):
        from lag.sandbox import SandboxedProvider

        console.post("/api/run", json={"prompt": "hi"})
        assert not isinstance(self.used[-1], SandboxedProvider)

    def test_one_container_serves_many_phone_chats(self):
        key, _ = _paired()
        for _ in range(3):
            handset.post("/api/run", json={"prompt": "hi"}, headers=_auth(key))
        assert FakeBox.started == 1

    def test_a_dead_container_is_replaced(self):
        key, _ = _paired()
        handset.post("/api/run", json={"prompt": "hi"}, headers=_auth(key))
        first = web._phone_box["box"]
        first.running = False
        web._phone_box["checked"] = 0                # due for a check
        handset.post("/api/run", json={"prompt": "hi"}, headers=_auth(key))
        assert first.stopped and web._phone_box["box"] is not first

    def test_no_docker_means_no_phone_chat(self, monkeypatch):
        from lag import sandbox

        monkeypatch.setattr(sandbox, "available", lambda: (False, "Docker is not running"))
        key, _ = _paired()
        r = handset.post("/api/run", json={"prompt": "hi"}, headers=_auth(key)).json()
        assert "sandbox" in r["error"] and "Docker is not running" in r["error"]
        assert not self.used                        # never ran unconfined

    def test_a_streamed_phone_chat_fails_closed_too(self, monkeypatch):
        from lag import sandbox

        monkeypatch.setattr(sandbox, "available", lambda: (False, "Docker is not running"))
        key, _ = _paired()
        body = handset.post("/api/stream", json={"prompt": "hi"}, headers=_auth(key)).text
        events = [json.loads(x[6:]) for x in body.splitlines() if x.startswith("data: ")]
        assert events[-1]["type"] == "error" and "sandbox" in events[-1]["message"]
        assert not self.used


# ── the CLI ───────────────────────────────────────────────────────────────────

class TestPhoneCommands:
    def test_pair_shows_a_code_that_works(self, monkeypatch):
        from typer.testing import CliRunner

        from lag import cli

        monkeypatch.setattr(cli, "_tailnet_url", lambda: "https://mac.tail1.ts.net")
        r = CliRunner().invoke(cli.app, ["phone", "pair"])
        assert r.exit_code == 0, r.output
        assert "https://mac.tail1.ts.net" in r.output
        shown = next(w for w in r.output.split() if len(w) == 9 and w[4] == "-")
        assert not isinstance(phone.redeem(shown, "p"), str)

    def test_devices_and_revoke(self):
        from typer.testing import CliRunner

        from lag import cli

        _paired("Pixel 8")
        run = CliRunner().invoke
        assert "Pixel 8" in run(cli.app, ["phone", "devices"]).output
        assert run(cli.app, ["phone", "revoke", "Pixel 8"]).exit_code == 0
        assert run(cli.app, ["phone", "revoke", "Pixel 8"]).exit_code == 1
        assert "no paired phones" in run(cli.app, ["phone", "devices"]).output

    def test_the_phone_port_must_differ_from_the_console(self):
        from typer.testing import CliRunner

        from lag import cli

        r = CliRunner().invoke(cli.app, ["serve", "--port", "8799", "--phone-port", "8799"])
        assert r.exit_code == 1
