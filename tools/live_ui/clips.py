# SPDX-License-Identifier: Apache-2.0
"""Record the demo clips from the real UI: a headless Chromium screen recording, converted to mp4 (h264, <= 20 s,
<= 720 px wide, no audio).

    python tools/live_ui/clips.py first_view --width 390 --out clips/          # phone width
    python tools/live_ui/clips.py first_view --width 1280 --out clips/         # desktop

Each scenario is a short script of taps and waits on the live page; it may call the agent API to issue commands
the UI then shows. Scenarios: see SCENARIOS at the bottom.
"""
from __future__ import annotations

import argparse
import asyncio
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from playwright.async_api import Page, async_playwright

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from alloy.live.agent import LiveAPI  # noqa: E402

URL = "http://127.0.0.1:47200/"


async def tap(page: Page, selector: str) -> None:
    """Tap through the DOM (the recording shows the page's response; there is no cursor to show)."""
    ok = await page.evaluate("(s) => { const e = document.querySelector(s); if (!e) return false; e.click(); return true; }",
                             selector)
    if not ok:
        raise RuntimeError(f"nothing to tap at {selector}")


async def pick_world(page: Page) -> int:
    """A world worth opening: the best auto-flagged one if there is one, else the first."""
    return int(await page.evaluate(
        "() => { const S = window.__live.S; const f = Object.keys(S.flags || {}).map(Number);"
        " const best = f.find((w) => (S.flags[w] || []).includes('best')); return best ?? f[0] ?? 0; }"))


async def first_view(page: Page, api: LiveAPI, wide: bool) -> None:
    await page.wait_for_timeout(3500)                       # the overview: every world of both machines
    w = await pick_world(page)
    await tap(page, f'.sq[data-w="{w}"]')
    await page.wait_for_timeout(6000)                       # live policy camera (and the 3D view on desktop)
    if not wide:
        await tap(page, "#toggle")                         # the 3D view on a phone
        await page.wait_for_timeout(5000)
        await tap(page, "#toggle")
        await page.wait_for_timeout(1500)
    else:
        await page.wait_for_timeout(4000)
    await tap(page, "#back")
    await page.wait_for_timeout(2000)


async def overview(page: Page, api: LiveAPI, wide: bool) -> None:
    await page.wait_for_timeout(16000)                      # every world of both machines, live


async def open_world(page: Page, wide: bool, world: int | None = None) -> int:
    w = await pick_world(page) if world is None else world
    await tap(page, f'.sq[data-w="{w}"]')
    await page.wait_for_timeout(2500)
    if not wide:
        await tap(page, "#toggle")                          # phone: show the 3D view for 3D scenarios
    return w


async def open_world_clip(page: Page, api: LiveAPI, wide: bool) -> None:
    await page.wait_for_timeout(2500)
    w = await pick_world(page)
    await tap(page, f'.sq[data-w="{w}"]')
    await page.wait_for_timeout(9000 if wide else 5000)
    if not wide:
        await tap(page, "#toggle")
        await page.wait_for_timeout(5000)


async def drag(page: Page, x0: float, y0: float, dx: float, dy: float, ms: int = 700) -> None:
    await page.mouse.move(x0, y0)
    await page.mouse.down()
    n = 14
    for i in range(1, n + 1):
        await page.mouse.move(x0 + dx * i / n, y0 + dy * i / n)
        await page.wait_for_timeout(ms // n)
    await page.mouse.up()


async def push(page: Page, api: LiveAPI, wide: bool) -> None:
    await open_world(page, wide)
    await page.wait_for_timeout(2500)
    for body, dx, dy in (("cable/2", 10, -70), ("cube", 110, -30)):    # the soft cable bends; the cube is shoved
        await tap(page, '[data-cmd="push"]')
        await page.wait_for_timeout(600)
        at = await page.evaluate(f"() => window.__live.view3d.screenOf('{body}')")
        await drag(page, at["x"], at["y"], dx, dy)
        await page.wait_for_timeout(5000)


async def setting(page: Page, api: LiveAPI, wide: bool) -> None:
    await open_world(page, wide)
    await page.wait_for_timeout(2500)
    await tap(page, '[data-cmd="settings"]')
    await page.wait_for_timeout(1500)
    # drag the "Reward: rise" slider to zero, as a user would
    box = await page.evaluate("""() => { const rows = [...document.querySelectorAll('#sheetBody .row')];
        const r = rows.find((x) => x.textContent.includes('rise')); const i = r.querySelector('input');
        const b = i.getBoundingClientRect(); return {x: b.left, y: b.top + b.height / 2, w: b.width, v: +i.value}; }""")
    x0 = box["x"] + box["w"] * box["v"] / 1000
    await drag(page, x0, box["y"], box["x"] + 2 - x0, 0, 900)
    await page.wait_for_timeout(1200)
    await page.mouse.click(10, 10)                          # close the sheet (tap the scrim)
    await page.evaluate("() => document.getElementById('scrim').click()")
    await page.wait_for_timeout(13000)                      # the reward line drops within two episodes


async def fork(page: Page, api: LiveAPI, wide: bool) -> None:
    w = await open_world(page, wide)
    await page.wait_for_timeout(2500)
    await tap(page, '[data-cmd="fork"]')
    await page.wait_for_timeout(1500)
    target = await page.evaluate("() => { const b = document.querySelector('#sheetBody .choice');"
                                 " return +b.textContent.match(/world (\\d+)/)[1] - 1; }")
    await tap(page, "#sheetBody .choice")
    await page.wait_for_timeout(2500)
    await tap(page, "#back")
    await page.wait_for_timeout(1500)
    await tap(page, f'.sq[data-w="{target}"]')              # the branch: same moment as the source
    await page.wait_for_timeout(500)
    if not wide:
        await tap(page, "#toggle")
    await page.wait_for_timeout(6000)


async def pause(page: Page, api: LiveAPI, wide: bool) -> None:
    await open_world(page, wide)
    await page.wait_for_timeout(3000)
    await tap(page, '[data-cmd="pause"]')
    await page.wait_for_timeout(5000)                       # video and 3D hold still
    await tap(page, '[data-cmd="pause"]')
    await page.wait_for_timeout(3500)
    await tap(page, "#back")
    await page.wait_for_timeout(1000)
    await tap(page, "#runPause")                            # the whole run
    await page.wait_for_timeout(3500)
    await tap(page, "#runPause")
    await page.wait_for_timeout(2500)


async def scrub(page: Page, api: LiveAPI, wide: bool) -> None:
    await open_world(page, wide)
    await page.wait_for_timeout(3000)
    box = await page.evaluate("() => { const b = document.getElementById('scrub').getBoundingClientRect();"
                              " return {x: b.left, y: b.top + b.height / 2, w: b.width}; }")
    x1 = box["x"] + box["w"] - 9
    await page.mouse.move(x1, box["y"])
    await page.mouse.down()
    for i in range(1, 25):                                  # drag back through the recording
        await page.mouse.move(x1 - box["w"] * 0.55 * i / 24, box["y"])
        await page.wait_for_timeout(120)
    await page.mouse.up()                                   # release: plays on from that moment
    await page.wait_for_timeout(7000)
    await tap(page, "#live")
    await page.wait_for_timeout(3000)


async def agent(page: Page, api: LiveAPI, wide: bool) -> None:
    w = await open_world(page, wide)
    await page.wait_for_timeout(3000)
    api.push(world=w, body="cube", force=(0.0, -3.0, 0.0), steps=10)     # an agent, through the HTTP API
    await page.wait_for_timeout(3500)
    api.pause(world=w)
    await page.wait_for_timeout(2500)
    api.resume(world=w)
    await page.wait_for_timeout(2000)
    await tap(page, "#logline")                             # the command log: issuer, step, result
    await page.wait_for_timeout(4000)


SCENARIOS = {"overview": overview, "first_view": first_view, "open_world": open_world_clip, "push": push, "setting": setting,
             "fork": fork, "pause": pause, "scrub": scrub, "agent": agent}


class Screencast:
    """DevTools screencast at the device pixel ratio (Playwright's own recorder drops it): JPEG frames with
    timestamps, assembled into a constant-rate mp4 by ffmpeg's concat demuxer."""

    def __init__(self, page: Page, size: tuple[int, int], tmp: Path) -> None:
        self.page, self.size, self.tmp = page, size, tmp
        self.frames: list[tuple[float, Path]] = []

    async def start(self) -> None:
        self.cdp = await self.page.context.new_cdp_session(self.page)
        self.cdp.on("Page.screencastFrame", lambda e: asyncio.ensure_future(self._frame(e)))
        await self.cdp.send("Page.startScreencast", {"format": "jpeg", "quality": 90, "maxWidth": self.size[0],
                                                     "maxHeight": self.size[1], "everyNthFrame": 1})

    async def _frame(self, e: dict) -> None:
        import base64  # noqa: PLC0415

        path = self.tmp / f"f{len(self.frames):05d}.jpg"
        path.write_bytes(base64.b64decode(e["data"]))
        self.frames.append((float(e["metadata"]["timestamp"]), path))
        try:
            await self.cdp.send("Page.screencastFrameAck", {"sessionId": e["sessionId"]})
        except Exception:  # noqa: BLE001 - the page may be closing
            pass

    async def stop(self) -> None:
        await self.cdp.send("Page.stopScreencast")

    def write(self, out: Path, end: float) -> None:
        lines = []
        for i, (t, path) in enumerate(self.frames):
            nxt = self.frames[i + 1][0] if i + 1 < len(self.frames) else end
            lines += [f"file '{path}'", f"duration {max(nxt - t, 0.001):.4f}"]
        lines.append(f"file '{self.frames[-1][1]}'")
        (self.tmp / "list.txt").write_text("\n".join(lines) + "\n")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i",
                        str(self.tmp / "list.txt"), "-t", "20", "-vf",
                        "scale='min(720,iw)':-2:flags=lanczos,fps=24,format=yuv420p", "-an", "-c:v", "libx264",
                        "-crf", "22", "-preset", "slow", "-movflags", "+faststart", str(out)], check=True)


async def record(name: str, width: int, out_dir: Path, url: str) -> Path:
    wide = width >= 700
    height = 900 if wide else 844
    scale = 1 if wide else 2
    tmp = Path(tempfile.mkdtemp(prefix=f"clip_{name}_"))
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{name}_{'desktop' if wide else 'phone'}.mp4"
    async with async_playwright() as p:
        browser = await p.chromium.launch(args=["--use-gl=swiftshader", "--enable-unsafe-swiftshader",
                                                "--ignore-gpu-blocklist", f"--force-device-scale-factor={scale}"])
        ctx = await browser.new_context(viewport={"width": width, "height": height}, device_scale_factor=scale,
                                        is_mobile=not wide, has_touch=not wide)
        page = await ctx.new_page()
        await page.goto(url)
        await page.wait_for_function("() => window.__live && window.__live.S.ov")
        await page.wait_for_timeout(800)
        cast = Screencast(page, (width * scale, height * scale), tmp)
        await cast.start()
        t0 = time.time()
        await SCENARIOS[name](page, LiveAPI(url, issuer="agent:demo"), wide)
        print(f"{name}: scenario took {time.time() - t0:.1f} s, {len(cast.frames)} frames")
        await cast.stop()
        end = cast.frames[-1][0] + 0.5 if cast.frames else 0.0
        await ctx.close()
        await browser.close()
    cast.write(out, end)
    shutil.rmtree(tmp, ignore_errors=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("scenario", choices=sorted(SCENARIOS))
    ap.add_argument("--width", type=int, default=390)
    ap.add_argument("--out", default="clips")
    ap.add_argument("--url", default=URL)
    a = ap.parse_args()
    print(asyncio.run(record(a.scenario, a.width, Path(a.out), a.url)))
