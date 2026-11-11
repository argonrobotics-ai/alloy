# SPDX-License-Identifier: Apache-2.0
"""The live UI's access check and auto-stop rule (tools/live_ui): no GPU, no network beyond localhost.
Owner names here are placeholders; the real ones come from ALLOY_LIVE_LOGIN / ALLOY_LIVE_TAILNET."""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "live_ui"))

from autostop import ViewerClock, stop_reason  # noqa: E402
from hub_access import AccessConfigError, AccessRule, access_middleware, allowed, host_name, rule_from_env  # noqa: E402

RULE = AccessRule(login="owner@example.com", host_suffix=".tailnet-example.ts.net")
OK = {"Tailscale-User-Login": "owner@example.com", "Host": "laptop.tailnet-example.ts.net"}


def test_allowed_owner_on_tailnet_host_with_or_without_port():
    assert allowed(OK, RULE)
    assert allowed({**OK, "Host": "laptop.tailnet-example.ts.net:443"}, RULE)
    assert allowed({**OK, "Host": "LAPTOP.TAILNET-EXAMPLE.TS.NET:8443"}, RULE)
    assert allowed({**OK, "Tailscale-User-Login": " Owner@Example.com "}, RULE)


def test_refused_without_or_with_wrong_identity():
    assert not allowed({"Host": OK["Host"]}, RULE)
    assert not allowed({**OK, "Tailscale-User-Login": "someone@example.com"}, RULE)
    assert not allowed({**OK, "Tailscale-User-Login": ""}, RULE)


def test_refused_off_tailnet_hosts():
    for host in ("127.0.0.1:47300", "localhost", "tailnet-example.ts.net", ".tailnet-example.ts.net",
                 "evil.tailnet-example.ts.net.attacker.com", "laptop.other-tailnet.ts.net", "", "[::1]:47300"):
        assert not allowed({**OK, "Host": host}, RULE), host


def test_rule_comes_from_env_and_refuses_when_unset():
    rule = rule_from_env({"ALLOY_LIVE_LOGIN": "owner@example.com", "ALLOY_LIVE_TAILNET": "tailnet-example.ts.net"})
    assert rule == RULE
    assert rule_from_env({"ALLOY_LIVE_LOGIN": "owner@example.com", "ALLOY_LIVE_TAILNET": ".tailnet-example.ts.net"}) == RULE
    for env in ({}, {"ALLOY_LIVE_LOGIN": "owner@example.com"}, {"ALLOY_LIVE_TAILNET": "tailnet-example.ts.net"},
                {"ALLOY_LIVE_LOGIN": " ", "ALLOY_LIVE_TAILNET": "tailnet-example.ts.net"}):
        with pytest.raises(AccessConfigError):
            rule_from_env(env)


def test_host_name_strips_port():
    assert host_name("a.b.ts.net:443") == "a.b.ts.net"
    assert host_name("[::1]:80") == "::1"
    assert host_name("a.b.ts.net") == "a.b.ts.net"


async def _client():
    from aiohttp import web  # noqa: PLC0415
    from aiohttp.test_utils import TestClient, TestServer  # noqa: PLC0415

    app = web.Application(middlewares=[access_middleware(RULE)])
    async def page(request):
        return web.Response(text="page")
    app.router.add_get("/", page)

    async def ws(request):
        sock = web.WebSocketResponse()
        await sock.prepare(request)
        await sock.send_str("hello")
        await sock.close()
        return sock
    app.router.add_get("/ws", ws)
    client = TestClient(TestServer(app))
    await client.start_server()
    return client


def test_middleware_page_and_websocket():
    aiohttp = pytest.importorskip("aiohttp")

    async def run() -> None:
        c = await _client()
        try:
            assert (await c.get("/", headers=OK)).status == 200
            assert (await c.get("/")).status == 403                       # no identity, Host is 127.0.0.1:port
            assert (await c.get("/", headers={**OK, "Tailscale-User-Login": "x@example.org"})).status == 403
            sock = await c.ws_connect("/ws", headers=OK)
            assert (await sock.receive_str()) == "hello"
            await sock.close()
            try:
                await c.ws_connect("/ws", headers={"Host": OK["Host"]})
                raise AssertionError("websocket without identity was admitted")
            except aiohttp.WSServerHandshakeError as e:
                assert e.status == 403
        finally:
            await c.close()
    asyncio.run(run())


def test_autostop_hard_limit_and_idle_after_first_viewer():
    t0, h, idle = 1000.0, 4 * 3600.0, 3600.0
    assert stop_reason(t0 + 10, t0, None, h, idle) is None
    assert stop_reason(t0 + h, t0, None, h, idle).startswith("hard")
    never = {"t": t0 + 3 * 3600, "viewers": 0, "first_viewer": None, "last_viewer": None}
    assert stop_reason(t0 + 3 * 3600, t0, never, h, idle) is None                 # no viewer yet: not counted
    clock = ViewerClock()
    clock.connect(t0 + 100)
    watching = clock.to_json(t0 + 5000)
    assert stop_reason(t0 + 5000, t0, watching, h, idle) is None
    clock.disconnect(t0 + 5000)
    left = clock.to_json(t0 + 5000)
    assert stop_reason(t0 + 5000 + idle - 1, t0, {**left, "t": t0 + 8000}, h, idle) is None
    assert stop_reason(t0 + 5000 + idle, t0, {**left, "t": t0 + 8600}, h, idle).startswith("idle")
    silent = {"t": t0 + 200, "viewers": 1, "first_viewer": t0 + 100, "last_viewer": t0 + 200}
    assert stop_reason(t0 + 200 + idle, t0, silent, h, idle).startswith("idle")   # a dead hub has no viewers
