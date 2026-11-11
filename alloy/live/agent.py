# SPDX-License-Identifier: Apache-2.0
"""Steer a live run from Python: the same typed commands the UI sends, through the hub's HTTP API (stdlib only).

    from alloy.live.agent import LiveAPI
    api = LiveAPI("http://127.0.0.1:47200", issuer="agent:tuner")
    ov = api.overview()                          # run, steps/s, reward line, one status per world, settings
    api.set("action_noise", 0.5)                 # run-wide setting, applied by every worker at its next step
    api.push(world=3, body="cube", force=(0, 4, 0), steps=15)
    api.fork(world=3, into=10)                   # world 10 continues from world 3's state
    api.pause(world=3); api.resume(world=3); api.reset(world=3)
    api.log(n=20)

Every call returns the workers' command-log entries (op, args, step applied, result) or raises ``LiveAPIError``.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any


class LiveAPIError(RuntimeError):
    """The hub refused the command (unknown op, bad world, setting out of range) or did not answer."""


class LiveAPI:
    def __init__(self, url: str = "http://127.0.0.1:47200", issuer: str = "agent", timeout: float = 30.0) -> None:
        self.url, self.issuer, self.timeout = url.rstrip("/"), issuer, timeout

    # --- reading ---------------------------------------------------------------------------------------------------

    def overview(self) -> dict[str, Any]:
        return self._get("/api/overview")

    def world(self, world: int) -> dict[str, Any]:
        return self._get(f"/api/world/{int(world)}")

    def log(self, n: int = 50) -> list[dict[str, Any]]:
        return self._get(f"/api/log?n={int(n)}")

    def ops(self) -> dict[str, Any]:
        """Every op with its arguments, and the whitelisted settings with their ranges."""
        return self._get("/api/ops")

    # --- commands --------------------------------------------------------------------------------------------------

    def command(self, op: str, **args: Any) -> list[dict[str, Any]]:
        out = self._post("/api/command", {"op": op, "args": args, "issuer": self.issuer})
        if not out.get("ok"):
            raise LiveAPIError(out.get("error") or json.dumps(out.get("entries")))
        return out["entries"]

    def pause(self, world: int | None = None) -> list[dict[str, Any]]:
        return self.command("pause", world=world)

    def resume(self, world: int | None = None) -> list[dict[str, Any]]:
        return self.command("resume", world=world)

    def set(self, name: str, value: float) -> list[dict[str, Any]]:
        return self.command("set", name=name, value=value)

    def push(self, world: int, body: str, force: tuple[float, float, float], steps: int = 15) -> list[dict[str, Any]]:
        return self.command("push", world=world, body=body, force=list(force), steps=steps)

    def fork(self, world: int, into: int | str) -> list[dict[str, Any]]:
        """``into``: a world index on the same machine, or "snapshot:<name>"."""
        return self.command("fork", world=world, into=into)

    def restore(self, snapshot: str, world: int) -> list[dict[str, Any]]:
        return self.command("restore", snapshot=snapshot, world=world)

    def reset(self, world: int) -> list[dict[str, Any]]:
        return self.command("reset", world=world)

    # --- http ------------------------------------------------------------------------------------------------------

    def _get(self, path: str) -> Any:
        return self._call(urllib.request.Request(self.url + path))

    def _post(self, path: str, body: dict[str, Any]) -> Any:
        req = urllib.request.Request(self.url + path, data=json.dumps(body).encode(), method="POST",
                                     headers={"Content-Type": "application/json"})
        return self._call(req)

    def _call(self, req: urllib.request.Request) -> Any:
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            try:
                detail = json.loads(e.read()).get("error")
            except (ValueError, AttributeError):
                detail = str(e)
            raise LiveAPIError(detail) from e
        except (urllib.error.URLError, TimeoutError) as e:
            raise LiveAPIError(f"hub at {self.url} did not answer: {e}") from e
