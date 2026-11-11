# SPDX-License-Identifier: Apache-2.0
"""Who may open the UI port: requests that come through the owner's Tailscale serve proxy.

Tailscale serve (https, tailnet only) forwards to the UI port with the caller's identity in ``Tailscale-User-Login``
and the tailnet host name in ``Host``. The UI port binds 127.0.0.1 and is reached only through an SSH tunnel from the
machine running ``tailscale serve``. Every UI request (the page, static files, the WebSocket upgrade) must carry the
owner's login and a Host under the tailnet's domain (a port is allowed); anything else gets 403.

Who the owner is comes from the environment, never from the code:
- ``ALLOY_LIVE_LOGIN``: the Tailscale login allowed in (an email address);
- ``ALLOY_LIVE_TAILNET``: the tailnet's DNS suffix, e.g. ``example-tailnet.ts.net``.
The hub refuses to open the UI port when either is unset (:func:`rule_from_env` raises).

Example::

    rule = AccessRule(login="owner@example.com", host_suffix=".tailnet-example.ts.net")
    allowed({"Tailscale-User-Login": "owner@example.com", "Host": "mac.tailnet-example.ts.net:443"}, rule)  # True
"""
from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass

ENV_LOGIN, ENV_TAILNET = "ALLOY_LIVE_LOGIN", "ALLOY_LIVE_TAILNET"


class AccessConfigError(RuntimeError):
    """The owner login or tailnet suffix is not configured."""


@dataclass(frozen=True)
class AccessRule:
    login: str
    host_suffix: str                     # with its leading dot: ".tailnet-example.ts.net"


def rule_from_env(env: Mapping[str, str] | None = None) -> AccessRule:
    """The rule from ``ALLOY_LIVE_LOGIN`` and ``ALLOY_LIVE_TAILNET``; raises when either is unset or empty."""
    env = os.environ if env is None else env
    login, tailnet = (env.get(ENV_LOGIN) or "").strip(), (env.get(ENV_TAILNET) or "").strip().strip(".")
    if not login or not tailnet:
        raise AccessConfigError(f"set {ENV_LOGIN} (the Tailscale login to admit) and {ENV_TAILNET} (the tailnet's "
                                "DNS suffix) before opening the UI port")
    return AccessRule(login, "." + tailnet.lower())


def host_name(host: str) -> str:
    """The Host header without its port: "a.b.ts.net:443" -> "a.b.ts.net", "[::1]:80" -> "::1"."""
    h = host.strip().lower()
    if h.startswith("["):
        return h[1:h.find("]")] if "]" in h else h
    return h.rsplit(":", 1)[0] if h.count(":") == 1 else h


def allowed(headers: Mapping[str, str], rule: AccessRule) -> bool:
    """True when the request names the owner and a tailnet host. ``headers`` is case-insensitive in aiohttp; a
    plain dict is looked up by the canonical names."""
    login = (headers.get("Tailscale-User-Login") or "").strip().lower()
    host = host_name(headers.get("Host") or "")
    return login == rule.login.lower() and host.endswith(rule.host_suffix) and len(host) > len(rule.host_suffix)


def access_middleware(rule: AccessRule) -> Callable:
    """aiohttp middleware answering 403 to every request the rule does not admit (aiohttp imported here only)."""
    from aiohttp import web  # noqa: PLC0415

    @web.middleware
    async def check(request, handler):
        if not allowed(request.headers, rule):
            return web.Response(status=403, text="forbidden\n")
        return await handler(request)
    return check
