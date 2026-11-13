# SPDX-License-Identifier: Apache-2.0
"""A small HTTP server that runs the 3D generator for ``alloy.create`` on a GPU machine (see ``alloy.create.generate``
for the models and their licences, ``alloy.create.gen3d_local`` for the install).

    python -m alloy.create.serve                     # 127.0.0.1:8765; tunnel it, or --host 0.0.0.0 on a private net
    export ALLOY_GEN3D_URL=http://<gpu-host>:8765    # on the machine that runs alloy.create

``GET /health`` -> ``{"ok", "models", "token_required", "busy", "loaded"}`` (``loaded``: models already in GPU
memory; the first request loads them, 1-2 min); ``POST /generate`` with ``{"text": ...}`` or ``{"photo_b64": ...}`` (and
optional ``seed``) -> ``{"mesh_obj_b64", "image_png_b64", "models", "prompt", "seconds"}`` or ``{"error", "fix"}``.
One request at a time. With ``$ALLOY_GEN3D_TOKEN`` set, requests must carry ``Authorization: Bearer <token>``.

Example::

    >>> main(["--port", "8765"])                     # doctest: +SKIP
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

log = logging.getLogger("alloy.create")

MAX_BODY = 32 * 1024 * 1024
MAX_ERROR = 600                      # characters of an exception message sent back
_LOCK = threading.Lock()


def handle(body: dict[str, Any]) -> dict[str, Any]:
    """One generation request -> the JSON reply (errors are replies too)."""
    from alloy.create.gen3d_local import generate_local  # noqa: PLC0415
    from alloy.create.generate import MODELS  # noqa: PLC0415

    bad = request_problem(body)
    if bad is not None:
        return bad
    text, photo_b64, seed = body.get("text"), body.get("photo_b64"), int(body.get("seed", 0))
    with tempfile.TemporaryDirectory(prefix="alloy-gen-") as tmp, _LOCK:
        work = Path(tmp)
        photo = None
        if photo_b64 is not None:
            photo = work / "input.png"
            photo.write_bytes(base64.b64decode(photo_b64, validate=True))
        try:
            out = generate_local(photo=str(photo) if photo else None, text=text, work=work, seed=seed)
        except Exception as exc:  # noqa: BLE001 - the client gets the error as data
            log.exception("generation failed")
            return {"error": f"{type(exc).__name__}: {str(exc)[:MAX_ERROR]}", "fix": "see the server log"}
        image = Path(out["image"]) if out.get("image") else None
        return {"mesh_obj_b64": base64.b64encode(Path(out["mesh"]).read_bytes()).decode("ascii"),
                "image_png_b64": base64.b64encode(image.read_bytes()).decode("ascii") if image else None,
                "models": MODELS, "prompt": out.get("prompt"), "seconds": out.get("seconds")}


def request_problem(body: Any) -> dict[str, str] | None:
    """Why ``body`` is not a generation request, as an ``{"error", "fix"}`` reply; None when it is one."""
    example = '{"text": "a ceramic mug", "seed": 0}'
    if not isinstance(body, dict):
        return {"error": f"the body must be a JSON object, got {type(body).__name__}", "fix": f"send {example}"}
    text, photo_b64, seed = body.get("text"), body.get("photo_b64"), body.get("seed", 0)
    if (text is None) == (photo_b64 is None):
        return {"error": "send exactly one of text or photo_b64", "fix": f"send {example}"}
    if isinstance(seed, bool) or not isinstance(seed, int):
        return {"error": f"seed must be an integer, got {seed!r}", "fix": f"send {example}"}
    if text is not None and (not isinstance(text, str) or not text.strip()):
        return {"error": "text must be a non-empty string", "fix": f"send {example}"}
    if photo_b64 is not None:
        try:
            ok = isinstance(photo_b64, str) and bool(base64.b64decode(photo_b64, validate=True))
        except ValueError:
            ok = False
        if not ok:
            return {"error": "photo_b64 is not base64 image data", "fix": "send the image file's bytes, base64-encoded"}
    return None


def health(token: str) -> dict[str, Any]:
    """What ``GET /health`` says: up, the models and licences, whether a token is needed, whether a request is
    running, and which models are already loaded."""
    from alloy.create.gen3d_local import loaded_models  # noqa: PLC0415
    from alloy.create.generate import MODELS  # noqa: PLC0415

    return {"ok": True, "models": MODELS, "token_required": bool(token), "busy": _LOCK.locked(),
            "loaded": loaded_models()}


class _Handler(BaseHTTPRequestHandler):
    token: str | None = None            # set by main() from $ALLOY_GEN3D_TOKEN; None: read the environment

    def _reply(self, code: int, data: dict[str, Any]) -> None:
        raw = json.dumps(data).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _token(self) -> str:
        return self.token if self.token is not None else os.environ.get("ALLOY_GEN3D_TOKEN", "").strip()

    def _authorized(self) -> bool:
        token = self._token()
        return not token or self.headers.get("Authorization", "") == f"Bearer {token}"

    def do_GET(self) -> None:  # noqa: N802 - http.server API
        if self.path.rstrip("/") == "/health":
            self._reply(200, health(self._token()))
        else:
            self._reply(404, {"error": f"no route {self.path}", "fix": "GET /health or POST /generate"})

    def do_POST(self) -> None:  # noqa: N802 - http.server API
        if not self._authorized():
            self._reply(401, {"error": "bad or missing token", "fix": "send Authorization: Bearer $ALLOY_GEN3D_TOKEN"})
            return
        length = self.headers.get("Content-Length", "0").strip()
        n = int(length) if length.isdigit() else -1
        if self.path.rstrip("/") != "/generate" or not 0 < n <= MAX_BODY:
            self.close_connection = True                        # the body, if any, was not read
            self._reply(400, {"error": "POST /generate with a JSON body up to 32 MB and its Content-Length",
                              "fix": "see the module doc"})
            return
        try:
            body = json.loads(self.rfile.read(n).decode("utf-8"))
        except ValueError as exc:
            self._reply(400, {"error": f"bad JSON: {exc}", "fix": '{"text": "a ceramic mug"}'})
            return
        bad = request_problem(body)
        if bad is not None:
            self._reply(400, bad)
            return
        self._reply(200, handle(body))

    def log_message(self, fmt: str, *args: Any) -> None:
        log.info("serve: " + fmt, *args)


def main(argv: list[str] | None = None) -> int:
    """Serve until interrupted. Example: ``python -m alloy.create.serve --host 0.0.0.0 --port 8765``."""
    p = argparse.ArgumentParser(prog="python -m alloy.create.serve", description=__doc__.split("\n\n")[0])
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    _Handler.token = os.environ.get("ALLOY_GEN3D_TOKEN", "").strip()
    server = ThreadingHTTPServer((args.host, args.port), _Handler)
    log.info("alloy.create.serve on %s:%d", args.host, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
