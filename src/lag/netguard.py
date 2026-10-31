"""Which network destinations untrusted code may reach. Standard library only.

Two things reach the network on an agent's behalf, and both are steerable by whatever
the model last read — a prompt injection in a fetched page is the usual route:

  - `tools.http_fetch`, running in the harness process on the host;
  - the agent's container shell, whose only way out is the egress proxy (egress.py).

Both must apply the SAME rule, so it lives here once. Two copies would drift, and the
drift would be invisible: a fix to one leaves the other open while the tests for the
fixed one stay green. Standard library only, because egress.py runs this file inside a
bare python image with nothing installed.

**The rule is default-deny.** Only globally routable addresses pass. The named checks
exist so a refusal can say what kind of address it was; the `is_global` line is what
actually decides. A block-list of bad kinds missed 100.64.0.0/10 (CGNAT — every
Tailscale device lives there), which Python calls neither private nor global.

**IPv4 hidden in IPv6 is unwrapped.** IPv4-mapped, 6to4, Teredo and NAT64 forms each
carry an IPv4 address a gateway may route to, and Python judges only the outer one — it
calls 64:ff9b::7f00:1 global although it carries 127.0.0.1.

**vet() returns the address it checked**, so a caller can connect to exactly that and
never resolve again. Resolving once to check and again to connect is the DNS-rebinding
gap: a hostile DNS server answers public the first time and 127.0.0.1 the second.
"""
from __future__ import annotations

import ipaddress
import socket

CGNAT = ipaddress.ip_network("100.64.0.0/10")
NAT64 = ipaddress.ip_network("64:ff9b::/96")


def with_embedded(ip):
    """Any IPv4 address an IPv6 form carries inside it, then the address itself.

    Embedded first, so a refusal says "carries 127.0.0.1" rather than "is reserved".
    """
    if ip.version == 6:
        if ip.ipv4_mapped:
            yield ip.ipv4_mapped
        if ip.sixtofour:
            yield ip.sixtofour
        if ip.teredo:
            yield from ip.teredo                     # (server, client)
        if ip in NAT64:
            yield ipaddress.IPv4Address(int(ip) & 0xFFFFFFFF)
    yield ip


def non_public(ip) -> str | None:
    """Why this one address is off limits, or None if it is on the public internet."""
    if ip.is_loopback:    return "is loopback"
    if ip.is_link_local:  return "is link-local (includes cloud metadata)"
    if ip.version == 4 and ip in CGNAT:
        return "is in 100.64.0.0/10, shared CGNAT space (Tailscale tailnets live here)"
    if ip.is_private:     return "is a private address"
    if ip.is_reserved:    return "is reserved"
    if ip.is_multicast:   return "is multicast"
    if ip.is_unspecified: return "is unspecified"
    if not ip.is_global:  return "is not a public internet address"   # the actual rule
    return None


def address_reason(addr: str) -> str | None:
    """Why one literal address is refused, naming anything embedded in it."""
    outer = ipaddress.ip_address(addr.split("%")[0])        # drop a %zone suffix
    for ip in with_embedded(outer):
        why = non_public(ip)
        if why:
            return f"{ip} {why}" if ip == outer else f"{outer} carries {ip}, which {why}"
    return None


def vet(host: str, port: int = 0) -> tuple[str | None, str | None]:
    """Resolve `host` and check EVERY address it resolves to.

    Returns (address to connect to, None) or (None, why refused). Any non-public answer
    refuses the whole name: a name that resolves to one public and one private address
    is exactly how a rebinding attacker would hedge, so "pick the good one" is wrong.
    """
    if not host:
        return None, "no host"
    try:
        infos = socket.getaddrinfo(host, port or None, type=socket.SOCK_STREAM)
    except socket.gaierror as e:
        return None, f"could not resolve host ({e})"
    addrs = list(dict.fromkeys(info[4][0] for info in infos))
    for a in addrs:
        why = address_reason(a)
        if why:
            return None, why
    return (addrs[0], None) if addrs else (None, "host resolved to nothing")
