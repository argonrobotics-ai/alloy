# SPDX-License-Identifier: Apache-2.0
"""The live hub: workers dial in over TCP, browsers and agents talk HTTP + WebSocket; everything is recorded.

    python tools/live_ui/hub.py --run cable-lift --worker-addr <internal-ip>:47100 --http-addr 127.0.0.1:47200

- Worker port: frames of ``alloy.live.wire`` (numbers, 3D state, meshes, H.264, command results in; picks and
  commands out). Bind it to the internal address only: it needs no public port.
- HTTP: ``/`` the UI, ``/ws`` the browser stream, ``/api/*`` the agent API (``hub_api.py``), on 127.0.0.1.
- UI port (``--ui-addr``, optional): the page and its WebSocket only, behind the Tailscale access check
  (``hub_access.py``), for viewers reaching it through ``tailscale serve``. Connected viewers are counted in
  ``<root>/<run>/viewers.json`` for the auto-stop (``autostop.py``).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import struct
import sys
import time
from pathlib import Path
from typing import Any

from aiohttp import WSMsgType, web

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))          # the worktree, for alloy.live.wire (stdlib + numpy only)

from alloy.live.frames import inflate  # noqa: E402
from alloy.live.wire import Kind, decode_body, encode, unpack_arrays  # noqa: E402
from autostop import ViewerClock  # noqa: E402
from hub_access import AccessConfigError, access_middleware, rule_from_env  # noqa: E402
from hub_api import add_api_routes  # noqa: E402
from hub_record import Recorder  # noqa: E402
from hub_state import RunState, Worker  # noqa: E402
from hub_timeline import video_packet  # noqa: E402

log = logging.getLogger("hub")


async def index(request: web.Request) -> web.FileResponse:
    return web.FileResponse(HERE / "static" / "index.html")


class Hub:
    def __init__(self, run: str, root: Path, load_picks: int = 0) -> None:
        self.state = RunState(run, load_picks)
        self.rec = Recorder(root / run)
        self.browsers: dict[web.WebSocketResponse, dict[str, Any]] = {}
        self.gop: dict[str, list[bytes]] = {}               # packets since the last key frame, per worker stream
        self.pending: dict[str, asyncio.Future] = {}        # command id -> future of its log entry
        self.log_rows: list[dict[str, Any]] = []
        self.topology: dict[int, dict[str, Any]] = self.rec.topologies()   # per global world, from the export
        self.viewers = ViewerClock()                         # remote (UI port) viewers, for the auto-stop
        self.viewers_file = self.rec.root / "viewers.json"
        self._load_log()

    def _load_log(self) -> None:
        p = self.rec.root / "commands.jsonl"
        if p.exists():
            self.log_rows = [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
            self.state.marks = [_mark(r) for r in self.log_rows]

    # --- workers ---------------------------------------------------------------------------------------------------

    async def worker_conn(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        peer = writer.get_extra_info("peername")
        w: Worker | None = None
        try:
            while True:
                (n,) = struct.unpack(">I", await reader.readexactly(4))
                kind, header, payload = decode_body(await reader.readexactly(n))
                if kind == Kind.HELLO:
                    if header.get("run") != self.state.run:
                        log.warning("worker %s is on run %r, not %r: refused", header.get("worker"), header.get("run"),
                                    self.state.run)
                        break
                    w = self.state.hello(header, writer)
                    log.info("worker %s from %s: %d worlds at offset %d", w.name, peer, w.worlds, w.offset)
                elif w is not None:
                    await self._on_frame(w, kind, header, payload)
        except (asyncio.IncompleteReadError, ConnectionError, ValueError) as e:
            log.info("worker %s disconnected (%s)", w.name if w else peer, type(e).__name__)
        finally:
            if w is not None and w.writer is writer:
                w.writer = None
            writer.close()

    async def _on_frame(self, w: Worker, kind: Kind, header: dict[str, Any], payload: memoryview) -> None:
        if kind == Kind.NUMBERS:
            arrays = unpack_arrays(header, payload)
            self.state.numbers(w, header, arrays)
            self.rec.numbers({"t": header["t"], "worker": w.name, "offset": w.offset, "step": header["step"],
                              "steps_per_s": header.get("steps_per_s"), "paused_run": header.get("paused_run"),
                              **{k: [round(float(x), 4) for x in v] for k, v in arrays.items()}})
        elif kind == Kind.STATE:
            self._on_state(w, header, unpack_arrays(header, payload))
        elif kind == Kind.VIDEO:
            self._on_video(w, header, bytes(payload))
        elif kind == Kind.RESULT:
            await self._on_result(w, header)

    def _on_state(self, w: Worker, header: dict[str, Any], arrays: dict[str, Any]) -> None:
        """Frames of the engine's frame export (one per picked world, ``alloy.live.frames``): the topology part is
        cached per world for viewers that open it later; the rest streams to the world's viewers and the recording."""
        t = float(header["t"])
        for raw in header["frames"]:
            frame = inflate(raw, arrays)
            gw = w.offset + int(frame.pop("world"))
            topo = frame.pop("topology", None)
            if topo is not None:
                self.topology[gw] = topo
                self.rec.topology(gw, t, topo)
                self._send_viewers(gw, json.dumps({"type": "geometry", "world": gw, "geo": topo}))
            frame["sim_t"] = frame.pop("t", None)              # "t" stays the hub's wall clock (timeline index)
            frame.update(type="state", world=gw, step=frame.get("period"))
            self.rec.state(gw, t, frame)
            self._send_viewers(gw, json.dumps({"t": t, **frame}))

    def _on_video(self, w: Worker, header: dict[str, Any], data: bytes) -> None:
        t, key = float(header["t"]), bool(header["key"])
        meta = {"tile": int(header["tile"]), "cols": int(header["cols"]),
                "tiles": [w.offset + int(x) for x in header["tiles"]]}
        self.rec.video(w.name, t, key, data, meta)
        pkt = video_packet(t, key, meta, data)
        if key:
            self.gop[w.name] = [pkt]
        elif w.name in self.gop:
            self.gop[w.name].append(pkt)
        tiles = set(meta["tiles"])
        for ws, sess in list(self.browsers.items()):
            if sess.get("world") in tiles and sess.get("live", True):
                asyncio.ensure_future(_send_bytes(ws, pkt))

    async def _on_result(self, w: Worker, header: dict[str, Any]) -> None:
        entry = dict(header["entry"])
        entry["worker"], entry["recv"] = w.name, time.time()
        if "world" in entry.get("args", {}) and entry["args"]["world"] is not None:
            entry["gworld"] = w.offset + int(entry["args"]["world"])
        w.hello["settings"] = header.get("settings", w.hello.get("settings"))
        self.rec.command(entry)
        self.log_rows.append(entry)
        self.state.marks.append(_mark(entry))
        fut = self.pending.pop(entry["id"], None)
        if fut is not None and not fut.done():
            fut.set_result(entry)
        self._broadcast(json.dumps({"type": "log", "entry": entry}))

    # --- commands ---------------------------------------------------------------------------------------------------

    async def command(self, op: str, args: dict[str, Any], issuer: str, wait: float = 20.0) -> list[dict[str, Any]]:
        """Route one command (global world ids) to the worker(s) that own it; returns their log entries."""
        from alloy.live.commands import make_command  # noqa: PLC0415

        targets = self._targets(op, args)
        futs = []
        for w, local_args in targets:
            cmd = make_command(op, local_args, issuer=issuer)
            fut = asyncio.get_running_loop().create_future()
            self.pending[cmd["id"]] = fut
            w.writer.write(encode(Kind.COMMAND, cmd))
            await w.writer.drain()
            futs.append(fut)
        done = await asyncio.wait_for(asyncio.gather(*futs), timeout=wait)
        return list(done)

    def _targets(self, op: str, args: dict[str, Any]) -> list[tuple[Worker, dict[str, Any]]]:
        connected = [w for w in self.state.workers.values() if w.connected]
        if not connected:
            raise ValueError("no worker is connected")
        if args.get("world") is None:
            if op not in ("pause", "resume", "set"):
                raise ValueError(f"{op} needs a world")
            return [(w, dict(args)) for w in connected]
        hit = self.state.owner(int(args["world"]))
        if hit is None or not hit[0].connected:
            raise ValueError(f"world {args['world']} has no connected worker")
        w, local = hit
        out = {**args, "world": local}
        if op == "fork" and not str(args["into"]).startswith("snapshot:"):
            dst = self.state.owner(int(args["into"]))
            if dst is None or dst[0] is not w:
                raise ValueError("fork copies within one machine: pick a slot on the same worker")
            out["into"] = dst[1]
        if op == "restore":
            out["snapshot"] = str(args["snapshot"])
        return [(w, out)]

    # --- browsers -------------------------------------------------------------------------------------------------

    def _broadcast(self, text: str) -> None:
        for ws in list(self.browsers):
            asyncio.ensure_future(_send_text(ws, text))

    def _send_viewers(self, gw: int, text: str) -> None:
        for ws, sess in list(self.browsers.items()):
            if sess.get("world") == gw and sess.get("live", True):
                asyncio.ensure_future(_send_text(ws, text))

    async def browser_remote(self, request: web.Request) -> web.WebSocketResponse:
        """A viewer on the UI port (already through the access check): counted while connected."""
        self.viewers.connect(time.time())
        log.info("viewer connected (%d now)", self.viewers.viewers)
        try:
            return await self.browser(request)
        finally:
            self.viewers.disconnect(time.time())
            log.info("viewer left (%d now)", self.viewers.viewers)

    async def browser(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse(heartbeat=20, max_msg_size=8 << 20)
        await ws.prepare(request)
        self.browsers[ws] = {"world": None, "live": True}
        await ws.send_str(json.dumps(self.state.overview() | {"flags": self.state.flags()}))
        await ws.send_str(json.dumps({"type": "log_all", "entries": self.log_rows[-200:]}))
        try:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    await self._browser_msg(ws, json.loads(msg.data))
        finally:
            self.browsers.pop(ws, None)
        return ws

    async def _browser_msg(self, ws: web.WebSocketResponse, m: dict[str, Any]) -> None:
        from hub_timeline import play, seek  # noqa: PLC0415

        sess = self.browsers[ws]
        if m["type"] in ("open", "close", "live", "seek", "play"):
            sess["play"] = None                                # any of these stops a running playback
        if m["type"] == "open":
            gw = int(m["world"])
            sess.update(world=gw, live=True)
            self.state.view(gw)
            hit = self.state.owner(gw)
            if gw in self.topology:
                await ws.send_str(json.dumps({"type": "geometry", "world": gw, "geo": self.topology[gw]}))
            for pkt in self.gop.get(hit[0].name, []) if hit else []:
                await ws.send_bytes(pkt)
        elif m["type"] == "close":
            sess.update(world=None, live=True)
        elif m["type"] == "live":
            sess["live"] = True
        elif m["type"] == "seek":
            sess["live"] = False
            await seek(self, ws, int(m["world"]) if m.get("world") is not None else None, float(m["t"]),
                       float(m.get("span", 0.0)))
        elif m["type"] == "play" and m.get("world") is not None:
            asyncio.ensure_future(play(self, ws, sess, int(m["world"]), float(m["t"])))
        elif m["type"] == "command":
            try:
                entries = await self.command(m["op"], m.get("args", {}), m.get("issuer", "user"))
                await ws.send_str(json.dumps({"type": "ack", "entries": entries}))
            except (ValueError, asyncio.TimeoutError) as e:
                await ws.send_str(json.dumps({"type": "error", "error": str(e) or "timeout"}))

    # --- periodic ---------------------------------------------------------------------------------------------------

    async def ticker(self) -> None:
        while True:
            await asyncio.sleep(1.0)
            try:
                self._tick(time.time())
            except Exception:  # noqa: BLE001 - the ticker must survive one bad frame
                log.exception("ticker")

    def _tick(self, now: float) -> None:
        self.viewers_file.write_text(json.dumps(self.viewers.to_json(now)))
        for gw, sess in [(s.get("world"), s) for s in self.browsers.values()]:
            if gw is not None:
                self.state.view(gw)
        self.state.tick_line(now)
        for w in self.state.workers.values():
            picks = self.state.picks(w, now)
            if w.connected and picks != w.picked:
                w.picked = picks
                w.writer.write(encode(Kind.PICK, {"worlds": picks}))
        self._broadcast(json.dumps(self.state.overview() | {"flags": self.state.flags()}))


def _mark(entry: dict[str, Any]) -> dict[str, Any]:
    return {"t": entry.get("wall"), "op": entry.get("op"), "world": entry.get("gworld"),
            "issuer": entry.get("issuer"), "ok": entry.get("ok")}


async def _send_text(ws: web.WebSocketResponse, text: str) -> None:
    try:
        await ws.send_str(text)
    except (ConnectionError, RuntimeError):
        pass


async def _send_bytes(ws: web.WebSocketResponse, data: bytes) -> None:
    try:
        await ws.send_bytes(data)
    except (ConnectionError, RuntimeError):
        pass


async def main(args: argparse.Namespace) -> None:
    rule = None
    if args.ui_addr:
        try:
            rule = rule_from_env()               # the UI port never opens without a configured owner
        except AccessConfigError as e:
            raise SystemExit(f"hub: refusing to open the UI port: {e}") from e
    hub = Hub(args.run, Path(args.root).expanduser(), args.load_picks)
    whost, wport = args.worker_addr.rsplit(":", 1)
    server = await asyncio.start_server(hub.worker_conn, whost, int(wport))
    app = web.Application()
    app["hub"] = hub
    app.router.add_get("/ws", hub.browser)
    add_api_routes(app)
    app.router.add_get("/", index)
    app.router.add_static("/static/", HERE / "static")
    runner = web.AppRunner(app)
    await runner.setup()
    hhost, hport = args.http_addr.rsplit(":", 1)
    await web.TCPSite(runner, hhost, int(hport)).start()
    log.info("hub %s: workers on %s, UI on http://%s", args.run, args.worker_addr, args.http_addr)
    if args.ui_addr:
        ui = web.Application(middlewares=[access_middleware(rule)])
        ui["hub"] = hub
        ui.router.add_get("/ws", hub.browser_remote)
        ui.router.add_get("/", index)
        ui.router.add_static("/static/", HERE / "static")
        ui_runner = web.AppRunner(ui)
        await ui_runner.setup()
        uhost, uport = args.ui_addr.rsplit(":", 1)
        await web.TCPSite(ui_runner, uhost, int(uport)).start()
        log.info("UI port for tailscale viewers on http://%s (access check on)", args.ui_addr)
    asyncio.ensure_future(hub.ticker())
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="towel-ppo")
    p.add_argument("--worker-addr", default="127.0.0.1:47100")
    p.add_argument("--http-addr", default="127.0.0.1:47200")
    p.add_argument("--root", default="live_recordings", help="where runs are recorded")
    p.add_argument("--load-picks", type=int, default=0, help="stream this many worlds per worker (load test)")
    p.add_argument("--ui-addr", default="", help="UI-only port behind the Tailscale access check, e.g. 127.0.0.1:47300 "
                   "(needs ALLOY_LIVE_LOGIN and ALLOY_LIVE_TAILNET)")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    asyncio.run(main(p.parse_args()))
