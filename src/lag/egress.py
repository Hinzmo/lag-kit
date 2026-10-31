"""The agent container's only way out: a forward proxy that applies netguard.

An agent's shell runs in a container on an `--internal` Docker network, which has no
route anywhere — not the internet, not the host, not the LAN. Measured, not assumed:
on that network host.docker.internal does not resolve, the Docker host's address is
"network unreachable", and the routing table holds only the local subnet. This proxy is
the one thing on that network that also sits on a routed one, so every byte that
leaves passes through `handle` below.

Why a proxy rather than firewall rules: the agent container runs with every capability
dropped, so nothing inside it can install rules — and anything that could install them
could remove them. The boundary has to sit outside what the agent controls. A proxy is
also the shape real tools already speak: pip, git (https), curl, wget and urllib all
honour HTTP(S)_PROXY, so they keep working, while anything that ignores the proxy (raw
sockets, ssh) simply has no route, which is the safe failure.

Two request shapes, because that is how HTTP proxying works:

  CONNECT host:443   HTTPS. The client asks for a raw tunnel and does TLS through it,
                     so the proxy never sees content — only the destination, which is
                     all it needs to judge.
  GET http://h/p     Plain HTTP. The client sends the absolute URL; the proxy forwards
                     it as an ordinary request.

For both, the destination is resolved ONCE, every address it resolves to is checked,
and the connection goes to the exact address that was checked. That closes DNS
rebinding for the shell: there is no second lookup for a hostile DNS server to answer
differently.

Standard library only, and runnable as a script: sandbox.py mounts this file and
netguard.py read-only into a bare python image and runs it there.
"""
from __future__ import annotations

import contextlib
import select
import socket
import socketserver
import sys
from urllib.parse import urlsplit

try:
    from . import netguard
except ImportError:          # run as a script inside the proxy container
    import netguard  # type: ignore[no-redef]

PORT = 3128
IDLE_TIMEOUT = 120           # seconds a tunnel may sit silent before it is closed
CONNECT_TIMEOUT = 10
MAX_HEAD = 64 * 1024         # a request head larger than this is refused, not buffered


def _log(msg: str) -> None:
    print(f"lag-egress: {msg}", file=sys.stderr, flush=True)


def _reply(sock: socket.socket, code: int, reason: str) -> None:
    """A short plain-text error the agent will see in curl/pip output."""
    status = {400: "Bad Request", 403: "Forbidden", 502: "Bad Gateway"}.get(code, "Error")
    body = f"blocked by the lag egress proxy: {reason}\n".encode()
    head = (f"HTTP/1.1 {code} {status}\r\nContent-Type: text/plain\r\n"
            f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n").encode()
    with contextlib.suppress(OSError):        # the client may already be gone
        sock.sendall(head + body)


def _read_head(sock: socket.socket) -> tuple[bytes, bytes] | None:
    """(request head, any bytes already read past it), or None if it never arrived."""
    buf = b""
    while b"\r\n\r\n" not in buf:
        if len(buf) > MAX_HEAD:
            return None
        chunk = sock.recv(8192)
        if not chunk:
            return None
        buf += chunk
    head, _, rest = buf.partition(b"\r\n\r\n")
    return head, rest


def _relay(a: socket.socket, b: socket.socket) -> None:
    """Copy bytes both ways until either side closes or the pair goes idle."""
    pair = {a: b, b: a}
    try:
        while True:
            ready, _, _ = select.select(list(pair), [], [], IDLE_TIMEOUT)
            if not ready:
                return
            for s in ready:
                data = s.recv(65536)
                if not data:
                    return
                pair[s].sendall(data)
    except OSError:
        return


class Handler(socketserver.BaseRequestHandler):
    # Indirection so tests can stand in a permissive check and prove the relay works
    # against a local server; production always uses netguard.vet.
    vet = staticmethod(netguard.vet)

    def handle(self) -> None:
        client: socket.socket = self.request
        client.settimeout(IDLE_TIMEOUT)
        try:
            got = _read_head(client)
            if got is None:
                _reply(client, 400, "incomplete or oversized request")
                return
            head, rest = got
            lines = head.decode("latin-1").split("\r\n")
            try:
                method, target, version = lines[0].split(" ", 2)
            except ValueError:
                _reply(client, 400, "malformed request line")
                return

            if method.upper() == "CONNECT":
                host, _, port_s = target.rpartition(":")
                host = host.strip("[]")
                if not host or not port_s.isdigit():
                    _reply(client, 400, f"bad CONNECT target {target!r}")
                    return
                upstream = self._open(client, host, int(port_s))
                if upstream is None:
                    return
                client.sendall(b"HTTP/1.1 200 Connection established\r\n\r\n")
                if rest:
                    upstream.sendall(rest)
            else:
                url = urlsplit(target)
                if url.scheme != "http" or not url.hostname:
                    _reply(client, 400, "send an absolute http:// URL, or CONNECT for https")
                    return
                upstream = self._open(client, url.hostname, url.port or 80)
                if upstream is None:
                    return
                path = (url.path or "/") + (f"?{url.query}" if url.query else "")
                # Hop-by-hop headers stay here; one request per upstream connection.
                keep = [h for h in lines[1:] if h and not h.lower().startswith(
                    ("proxy-", "connection:", "keep-alive:"))]
                req = "\r\n".join([f"{method} {path} {version}", *keep,
                                   "Connection: close", "", ""])
                upstream.sendall(req.encode("latin-1") + rest)

            with upstream:
                _relay(client, upstream)
        except OSError:
            return

    def _open(self, client: socket.socket, host: str, port: int) -> socket.socket | None:
        """Vet the destination, then connect to the exact address that was vetted."""
        addr, why = self.vet(host, port)
        if why:
            _log(f"refused {host}:{port} — {why}")
            _reply(client, 403, f"{host}: {why}")
            return None
        try:
            up = socket.create_connection((addr, port), timeout=CONNECT_TIMEOUT)
        except OSError as e:
            _reply(client, 502, f"could not connect to {host}:{port} ({e})")
            return None
        up.settimeout(IDLE_TIMEOUT)
        _log(f"allowed {host}:{port} -> {addr}")
        return up


class Server(socketserver.ThreadingTCPServer):
    daemon_threads = True
    allow_reuse_address = True


def main(host: str = "0.0.0.0", port: int = PORT) -> None:
    with Server((host, port), Handler) as srv:
        _log(f"listening on {host}:{port}")
        srv.serve_forever()


if __name__ == "__main__":
    main()
