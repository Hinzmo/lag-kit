"""Egress proxy tests — the proxy in-process, against a local upstream. No Docker.

Refusals run against the real netguard.vet. The pass-through tests stand in a
permissive check (a local upstream is, correctly, non-public) to prove the relay
itself works for both request shapes the proxy speaks.
"""
from __future__ import annotations

import http.server
import socket
import threading
import urllib.request

import pytest

from lag import egress


def _serve(server):
    threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.02},
                     daemon=True).start()
    return server


@pytest.fixture
def upstream():
    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            body = f"upstream saw {self.path} host={self.headers.get('Host')}".encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = _serve(http.server.HTTPServer(("127.0.0.1", 0), H))
    yield srv.server_address[1]
    srv.shutdown()
    srv.server_close()


def _proxy(vet=None):
    handler = egress.Handler
    if vet is not None:
        handler = type("H", (egress.Handler,), {"vet": staticmethod(vet)})
    return _serve(egress.Server(("127.0.0.1", 0), handler))


@pytest.fixture
def strict_proxy():
    srv = _proxy()
    yield srv.server_address[1]
    srv.shutdown()
    srv.server_close()


@pytest.fixture
def open_proxy():
    """Treats every name as public and pins it to 127.0.0.1 — see module docstring."""
    srv = _proxy(lambda host, port=0: ("127.0.0.1", None))
    yield srv.server_address[1]
    srv.shutdown()
    srv.server_close()


def _raw(port: int, data: bytes) -> bytes:
    with socket.create_connection(("127.0.0.1", port), timeout=5) as s:
        s.sendall(data)
        out = b""
        while chunk := s.recv(65536):
            out += chunk
        return out


def _via(proxy_port: int, url: str) -> str:
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({"http": f"http://127.0.0.1:{proxy_port}"}))
    try:
        with opener.open(url, timeout=5) as r:
            return r.read().decode()
    except urllib.error.HTTPError as e:
        return f"HTTP {e.code}: {e.read().decode()}"


class TestRefusals:
    """The real guard: the agent container's view of the host, LAN and tailnet."""

    @pytest.mark.parametrize("target", [
        "127.0.0.1:11434",          # the host's Ollama, as seen from Docker Desktop
        "192.168.65.254:11434",     # Docker Desktop's host address
        "192.168.1.1:80",           # a router
        "100.100.100.100:53",       # Tailscale MagicDNS
        "[::ffff:127.0.0.1]:22",
    ])
    def test_connect_to_a_non_public_address_is_refused(self, strict_proxy, target):
        out = _raw(strict_proxy, f"CONNECT {target} HTTP/1.1\r\nHost: {target}\r\n\r\n".encode())
        assert out.startswith(b"HTTP/1.1 403"), out
        assert b"blocked by the lag egress proxy" in out

    def test_plain_http_to_loopback_is_refused(self, strict_proxy, upstream):
        out = _via(strict_proxy, f"http://127.0.0.1:{upstream}/secret")
        assert out.startswith("HTTP 403") and "loopback" in out
        assert "upstream saw" not in out

    def test_localhost_by_name_is_refused(self, strict_proxy):
        out = _raw(strict_proxy, b"CONNECT localhost:443 HTTP/1.1\r\n\r\n")
        assert out.startswith(b"HTTP/1.1 403")

    def test_non_http_absolute_urls_are_refused(self, strict_proxy):
        out = _raw(strict_proxy, b"GET ftp://example.com/x HTTP/1.1\r\n\r\n")
        assert out.startswith(b"HTTP/1.1 400")

    def test_garbage_is_refused_not_crashed_on(self, strict_proxy):
        assert _raw(strict_proxy, b"nonsense\r\n\r\n").startswith(b"HTTP/1.1 400")
        # and the proxy still serves afterwards
        assert _raw(strict_proxy, b"CONNECT localhost:1 HTTP/1.1\r\n\r\n").startswith(b"HTTP/1.1 403")


class TestPassThrough:
    def test_plain_http_is_forwarded(self, open_proxy, upstream):
        out = _via(open_proxy, f"http://example.test:{upstream}/hello?x=1")
        assert out == f"upstream saw /hello?x=1 host=example.test:{upstream}"

    def test_connect_opens_a_tunnel(self, open_proxy, upstream):
        """What HTTPS does: CONNECT, then speak to the upstream through the tunnel."""
        out = _raw(open_proxy,
                   f"CONNECT example.test:{upstream} HTTP/1.1\r\n\r\n"
                   f"GET /tunnelled HTTP/1.0\r\nHost: example.test\r\n\r\n".encode())
        assert out.startswith(b"HTTP/1.1 200 Connection established")
        assert b"upstream saw /tunnelled" in out

    def test_it_connects_to_the_vetted_address_without_resolving_again(self, open_proxy, upstream):
        """The DNS-rebinding defence. `never-resolves.invalid` cannot be looked up at
        all, so reaching the upstream proves the proxy connected to the address vet()
        returned rather than resolving the name a second time."""
        out = _via(open_proxy, f"http://never-resolves.invalid:{upstream}/pinned")
        assert out.startswith("upstream saw /pinned")


class TestVet:
    def test_one_bad_answer_refuses_the_whole_name(self, monkeypatch):
        """A name resolving to one public and one private address is how a rebinding
        attacker hedges, so 'use the good one' would be the wrong rule."""
        from lag import netguard
        monkeypatch.setattr(netguard.socket, "getaddrinfo", lambda *a, **k: [
            (2, 1, 6, "", ("8.8.8.8", 0)), (2, 1, 6, "", ("10.0.0.5", 0))])
        addr, why = netguard.vet("mixed.example", 443)
        assert addr is None and "10.0.0.5" in why

    def test_a_public_name_returns_its_address(self, monkeypatch):
        from lag import netguard
        monkeypatch.setattr(netguard.socket, "getaddrinfo", lambda *a, **k: [
            (2, 1, 6, "", ("8.8.8.8", 0))])
        assert netguard.vet("dns.example", 443) == ("8.8.8.8", None)
