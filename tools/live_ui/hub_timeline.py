# SPDX-License-Identifier: Apache-2.0
"""Timeline seeks: what the UI shows for a past moment ``t``.

A seek sends the overview grid as it was at ``t`` (every world's numbers), and for an open world its 3D frame at
``t`` and the video packets from the key frame before ``t`` up to ``t + span`` (``span`` > 0 plays on from there).
"""
from __future__ import annotations

import json
import struct
from typing import Any

from aiohttp import web

from hub_state import _status

VIDEO_TYPE = 2
VIDEO_HEAD = struct.Struct(">BdBHBB")           # type 2, t, key, tile px, cols, number of tiles; then u32 worlds


def video_packet(t: float, key: bool, meta: dict[str, Any], data: bytes) -> bytes:
    """A mosaic packet for the browser: header, the global world of each tile, the Annex-B access unit."""
    tiles = meta["tiles"]
    return (VIDEO_HEAD.pack(VIDEO_TYPE, t, int(key), int(meta["tile"]), int(meta["cols"]), len(tiles))
            + struct.pack(f">{len(tiles)}I", *tiles) + data)


async def seek(hub: Any, ws: web.WebSocketResponse, gw: int | None, t: float, span: float) -> None:
    await _grid(hub, ws, t)
    if gw is None:
        return
    frame = hub.rec.state_at(gw, t)
    if frame is not None:
        await ws.send_str(json.dumps({**frame, "type": "state", "past": True}))
    stream = _stream(hub, gw)
    packets = hub.rec.video_window(stream, t, t + span) if stream else []
    await ws.send_str(json.dumps({"type": "past_video", "world": gw, "t": t, "count": len(packets)}))
    for pt, key, data, meta in packets:
        await ws.send_bytes(video_packet(pt, key, meta, data))


def _stream(hub: Any, gw: int) -> str | None:
    hit = hub.state.owner(gw)
    return hit[0].name if hit else None


async def play(hub: Any, ws: web.WebSocketResponse, sess: dict[str, Any], gw: int, t: float) -> None:
    """Replay world ``gw`` from ``t`` in real time (video, 3D, grid) until the session goes live or seeks again;
    at the end of the recording the browser is told to go live."""
    import asyncio  # noqa: PLC0415
    import time  # noqa: PLC0415

    token = object()
    sess.update(live=False, play=token)
    await seek(hub, ws, gw, t, 0.0)
    stream = _stream(hub, gw)
    vidx, sidx = hub.rec.video_idx.get(stream), hub.rec.state_idx.get(gw)
    vi = vidx.at(t) + 1 if vidx else 0
    si, last_grid, wall0 = (sidx.at(t) if sidx else -1), t, time.time()
    while sess.get("play") is token and not ws.closed:
        cur = t + (time.time() - wall0)
        span = hub.rec.span()
        if span is None or cur >= span[1] - 0.5:
            await ws.send_str(json.dumps({"type": "ended"}))
            break
        while vidx and vi < len(vidx.t) and vidx.t[vi] <= cur:
            data = hub.rec._read(f"video/{stream}.h264", vidx.off[vi], vidx.size[vi])
            await ws.send_bytes(video_packet(vidx.t[vi], vidx.key[vi], vidx.meta[vi], data))
            vi += 1
        if sidx:
            j = sidx.at(cur)
            if j > si:
                si = j
                frame = json.loads(hub.rec._read(f"state/{gw}.jsonl", sidx.off[j], sidx.size[j]))
                await ws.send_str(json.dumps({**frame, "type": "state", "play": True}))
        if cur - last_grid >= 1.0:
            last_grid = cur
            await _grid(hub, ws, cur)
        await asyncio.sleep(0.03)


async def _grid(hub: Any, ws: web.WebSocketResponse, t: float) -> None:
    import numpy as np  # noqa: PLC0415

    rows = {r["worker"]: r for r in hub.rec.numbers_at(t)}
    status, ret = [], []
    for w in sorted(hub.state.workers.values(), key=lambda x: x.offset):
        r = rows.get(w.name)
        nums = {k: np.asarray(r[k]) for k in ("success", "ep_success", "done", "paused", "health")} if r else {}
        status += _status(nums, w.worlds, bool(r))
        ret += r["ret"] if r else [0.0] * w.worlds
    await ws.send_str(json.dumps({"type": "past", "t": t, "status": status, "ret": ret}))
