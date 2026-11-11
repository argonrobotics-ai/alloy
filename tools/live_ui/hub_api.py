# SPDX-License-Identifier: Apache-2.0
"""The agent API: the same commands the UI sends, as plain HTTP + JSON (``alloy.live.agent.LiveAPI`` wraps it).

- ``GET  /api/overview``            run name, steps/s, reward line, one status per world, workers and settings
- ``GET  /api/world/<gw>``          one world's latest numbers, its worker, whether it streams in full
- ``GET  /api/log?n=50``            the newest command log entries
- ``GET  /api/ops``                 every command op and its arguments, and the whitelisted settings
- ``POST /api/command``             ``{"op": "set", "args": {...}, "issuer": "agent:tuner"}`` -> the log entries
- ``GET  /api/timeline``            the recorded time span and the command marks
"""
from __future__ import annotations

import asyncio
from typing import Any

from aiohttp import web


def add_api_routes(app: web.Application) -> None:
    app.router.add_get("/api/overview", overview)
    app.router.add_get("/api/world/{gw}", world)
    app.router.add_get("/api/log", command_log)
    app.router.add_get("/api/ops", ops)
    app.router.add_post("/api/command", command)
    app.router.add_get("/api/timeline", timeline)


def _hub(request: web.Request) -> Any:
    return request.app["hub"]


async def overview(request: web.Request) -> web.Response:
    hub = _hub(request)
    return web.json_response(hub.state.overview() | {"flags": hub.state.flags()})


async def world(request: web.Request) -> web.Response:
    hub, gw = _hub(request), int(request.match_info["gw"])
    hit = hub.state.owner(gw)
    if hit is None:
        return web.json_response({"error": f"no world {gw}"}, status=404)
    w, local = hit
    nums = {k: float(v[local]) for k, v in w.numbers.items()} if w.numbers else {}
    return web.json_response({"world": gw, "worker": w.name, "local": local, "numbers": nums,
                              "picked": local in w.picked, "flags": hub.state.flags().get(gw, []),
                              "bodies": [b["name"] for b in w.hello.get("bodies", [])]})


async def command_log(request: web.Request) -> web.Response:
    n = int(request.query.get("n", 50))
    return web.json_response(_hub(request).log_rows[-n:])


async def ops(request: web.Request) -> web.Response:
    from alloy.live.commands import OPS  # noqa: PLC0415

    hub = _hub(request)
    settings = next((w.hello.get("settings", []) for w in hub.state.workers.values()), [])
    return web.json_response({"ops": {k: list(v) for k, v in OPS.items()}, "settings": settings})


async def command(request: web.Request) -> web.Response:
    hub = _hub(request)
    try:
        body = await request.json()
        entries = await hub.command(str(body["op"]), dict(body.get("args") or {}), str(body.get("issuer", "agent")))
    except (ValueError, KeyError) as e:
        return web.json_response({"ok": False, "error": str(e)}, status=400)
    except asyncio.TimeoutError:
        return web.json_response({"ok": False, "error": "the worker did not apply it within 20 s"}, status=504)
    return web.json_response({"ok": all(e.get("ok") for e in entries), "entries": entries})


async def timeline(request: web.Request) -> web.Response:
    hub = _hub(request)
    return web.json_response({"span": hub.rec.span(), "marks": hub.state.marks[-500:],
                              "recorded_worlds": hub.rec.recorded_worlds()})
