"""Phone access: pairing codes, device keys, and the log of refused requests.

The phone reaches `lag serve` through `tailscale serve`, which forwards only to
127.0.0.1. Forwarded that way, a phone request arrives from 127.0.0.1 exactly as the
Mac's own browser does, so nothing about the connection says which is which. The phone
therefore gets a port of its own (`lag serve --phone-port`), and every request on that
port must carry a device key. The console port works as before.

A device key is handed out once, when a phone redeems a short-lived pairing code shown
on the Mac (`lag phone pair`), and only its SHA-256 is stored. A copied devices file
lets nobody in; revoking a phone is deleting its row.

The CLI and the server are separate processes that both edit these files. Every
read-modify-write takes an OS file lock, so a revoke from the CLI cannot be undone by
the server writing back a `last_seen` it read a moment earlier.
"""
from __future__ import annotations

import contextlib
import hashlib
import hmac
import json
import os
import secrets
import time
from pathlib import Path

try:
    import fcntl
except ImportError:  # Windows; the kit's sandbox needs a Unix host anyway
    fcntl = None

PHONE_DIR = Path("phone")
CODE_MINUTES = 5          # how long a pairing code works
CODE_TRIES = 5            # wrong guesses before the code dies
# Crockford's base32: no I, L, O or U, so a code read off a screen is never ambiguous.
# 8 characters = 40 bits; with 5 tries in 5 minutes, guessing is hopeless.
_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
_CODE_LEN = 8
_TOUCH_EVERY = 60         # seconds between last_seen writes for one device
_LOG_LIMIT = 1_000_000    # bytes before refused.log rolls over to refused.log.1


def _path(name: str) -> Path:
    return PHONE_DIR / name


def _hash(secret: str) -> str:
    return hashlib.sha256(secret.encode("utf-8")).hexdigest()


def _now() -> float:
    return time.time()


def _stamp(t: float | None = None) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(t or _now()))


@contextlib.contextmanager
def _locked():
    """Hold the phone folder's lock for one read-modify-write."""
    PHONE_DIR.mkdir(parents=True, exist_ok=True)
    with open(_path(".lock"), "a+") as f:
        if fcntl is not None:
            fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(f, fcntl.LOCK_UN)


def _read(name: str, default):
    try:
        return json.loads(_path(name).read_text())
    except (FileNotFoundError, ValueError):
        return default


def _write(name: str, data) -> None:
    """Atomic, and readable by this user only: the files hold hashes, not keys, but
    there is no reason for anyone else to read them."""
    PHONE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = _path(name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.chmod(tmp, 0o600)
    tmp.replace(_path(name))


# ── pairing codes ─────────────────────────────────────────────────────────────

def normalise_code(code: str) -> str:
    """What a person types, as the code was printed: case, dashes and spaces ignored,
    and the letters people mistake for digits read as those digits."""
    c = "".join(ch for ch in (code or "").upper() if ch.isalnum())
    return c.translate(str.maketrans({"O": "0", "I": "1", "L": "1"}))


def new_code(url: str | None = None, minutes: int = CODE_MINUTES) -> tuple[str, float]:
    """Start a pairing: returns (code, expiry time). Replaces any earlier code."""
    code = "".join(secrets.choice(_ALPHABET) for _ in range(_CODE_LEN))
    expires = _now() + minutes * 60
    with _locked():
        _write("pairing.json", {"code_sha256": _hash(code), "expires": expires,
                                "tries_left": CODE_TRIES, "url": url or ""})
    return code, expires


def show_code(code: str) -> str:
    return f"{code[:4]}-{code[4:]}"


def redeem(code: str, name: str, login: str = "") -> tuple[str, dict] | str:
    """Trade a pairing code for a device key: (key, device), or why not.

    The key is returned here and nowhere else; only its hash is kept. The code works
    once. Each wrong guess spends a try, and the last one deletes the code.
    """
    name = " ".join((name or "").split())[:40] or "phone"
    with _locked():
        p = _read("pairing.json", None)
        if not p:
            return "no pairing code is active; run `lag phone pair` on the Mac"
        if _now() > p.get("expires", 0):
            _path("pairing.json").unlink(missing_ok=True)
            return "that pairing code has expired; run `lag phone pair` again"
        if not hmac.compare_digest(_hash(normalise_code(code)), p.get("code_sha256", "")):
            p["tries_left"] = int(p.get("tries_left", 0)) - 1
            if p["tries_left"] <= 0:
                _path("pairing.json").unlink(missing_ok=True)
                return "wrong code, and no tries left; run `lag phone pair` again"
            _write("pairing.json", p)
            return f"wrong code ({p['tries_left']} tries left)"
        _path("pairing.json").unlink(missing_ok=True)

        devices = _read("devices.json", [])
        taken = {d["name"] for d in devices}
        unique, n = name, 2
        while unique in taken:
            unique, n = f"{name} ({n})", n + 1
        key = secrets.token_urlsafe(32)                     # 256 bits
        device = {"name": unique, "key_sha256": _hash(key), "paired": _stamp(),
                  "last_seen": _stamp(), "tailscale_login": login or ""}
        devices.append(device)
        _write("devices.json", devices)
    return key, public(device)


# ── devices ───────────────────────────────────────────────────────────────────

def public(device: dict) -> dict:
    """A device as anyone may see it: everything but the key hash."""
    return {k: v for k, v in device.items() if k != "key_sha256"}


def devices() -> list[dict]:
    return [public(d) for d in _read("devices.json", [])]


_last_touch: dict[str, float] = {}


def lookup(key: str) -> dict | None:
    """The paired device this key belongs to, or None. Read fresh every time, so a
    revoke from the CLI takes effect on the very next request."""
    if not key:
        return None
    h = _hash(key)
    for d in _read("devices.json", []):
        if hmac.compare_digest(h, d.get("key_sha256", "")):
            return d
    return None


def touch(device: dict) -> None:
    """Record that a device was just seen, at most once a minute per device."""
    name = device.get("name", "")
    if _now() - _last_touch.get(name, 0) < _TOUCH_EVERY:
        return
    _last_touch[name] = _now()
    with _locked():
        rows = _read("devices.json", [])
        for d in rows:
            if d.get("key_sha256") == device.get("key_sha256"):
                d["last_seen"] = _stamp()
        _write("devices.json", rows)


def revoke(name: str) -> bool:
    """Forget a device. True if there was one by that name."""
    with _locked():
        rows = _read("devices.json", [])
        kept = [d for d in rows if d.get("name") != name]
        if len(kept) == len(rows):
            return False
        _write("devices.json", kept)
    _last_touch.pop(name, None)
    return True


# ── the refused-request log ───────────────────────────────────────────────────

def log_refused(path: str, reason: str, login: str = "") -> None:
    """One line per refused phone request. Never the key: only what was asked for,
    why it was refused, and the Tailscale login when Serve supplied one."""
    with contextlib.suppress(OSError):
        PHONE_DIR.mkdir(parents=True, exist_ok=True)
        log = _path("refused.log")
        if log.exists() and log.stat().st_size > _LOG_LIMIT:
            log.replace(_path("refused.log.1"))
        with open(log, "a") as f:
            f.write(json.dumps({"at": _stamp(), "path": path, "reason": reason,
                                "tailscale_login": login}) + "\n")
