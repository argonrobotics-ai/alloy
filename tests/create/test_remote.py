# SPDX-License-Identifier: Apache-2.0
"""create's generator client: every way a generation server can fail gives an error that says what happened and
what to do (no token, wrong token, nothing listening, a dropped connection that a retry fixes, a machine that went
away mid-request, a request that times out), and /health says whether a token is needed and what is loaded."""

from __future__ import annotations

import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from alloy.create import remote
from alloy.create.request import CreateError

OK = {"mesh_obj_b64": "", "image_png_b64": None, "models": {}, "prompt": "x", "seconds": 0.1}


def _server(posts: list[str], token: str = ""):
    """A stand-in generation server: each POST takes the next behaviour of ``posts`` ("ok", "drop", "slow")."""
    plan = list(posts)

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, data: dict) -> None:
            raw = json.dumps(data).encode()
            self.send_response(code)
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self) -> None:  # noqa: N802
            self._send(200, {"ok": True, "token_required": bool(token), "busy": False, "loaded": []})

        def do_POST(self) -> None:  # noqa: N802
            self.rfile.read(int(self.headers.get("Content-Length", "0")))
            if token and self.headers.get("Authorization") != f"Bearer {token}":
                self._send(401, {"error": "bad or missing token", "fix": "send the token"})
                return
            what = plan.pop(0) if plan else "ok"
            if what == "drop":                                    # the process died mid-request
                self.close_connection = True
                self.connection.shutdown(socket.SHUT_RDWR)
                return
            if what == "slow":
                time.sleep(1.5)
            self._send(200, OK)

        def log_message(self, *args) -> None:
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, f"http://127.0.0.1:{httpd.server_address[1]}"


@pytest.fixture()
def serve():
    started = []

    def start(posts=(), token=""):
        httpd, url = _server(list(posts), token)
        started.append(httpd)
        return url

    yield start
    for httpd in started:
        httpd.shutdown()


def test_token_missing_or_wrong_says_set_the_token(serve, monkeypatch):
    url = serve(token="secret")
    monkeypatch.delenv(remote.TOKEN_ENV, raising=False)
    with pytest.raises(CreateError) as e:
        remote.post_generate(url, {"text": "a bowl"})
    assert "needs a token and none is set" in e.value.message and remote.TOKEN_ENV in e.value.fix
    monkeypatch.setenv(remote.TOKEN_ENV, "wrong")
    with pytest.raises(CreateError) as e:
        remote.post_generate(url, {"text": "a bowl"})
    assert "refused" in e.value.message and "not the server's" in e.value.message
    monkeypatch.setenv(remote.TOKEN_ENV, "secret")
    assert remote.post_generate(url, {"text": "a bowl"}) == OK


def test_nothing_listening_says_start_the_server():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]                                  # closed again when the block ends
    with pytest.raises(CreateError) as e:
        remote.post_generate(f"http://127.0.0.1:{port}", {"text": "a bowl"})
    assert "nothing is listening" in e.value.message and "alloy.create.serve" in e.value.fix


def test_a_dropped_connection_is_retried_once_when_the_server_still_answers(serve):
    assert remote.post_generate(serve(["drop", "ok"]), {"text": "a bowl"}) == OK
    with pytest.raises(CreateError) as e:
        remote.post_generate(serve(["drop", "drop"]), {"text": "a bowl"})
    assert "closed the connection twice" in e.value.message and "log" in e.value.fix


def test_a_machine_gone_mid_request_says_restart_it(serve, monkeypatch):
    url = serve(["drop"])
    monkeypatch.setattr(remote, "health_of", lambda base: None)    # preempted: /health stops answering
    with pytest.raises(CreateError) as e:
        remote.post_generate(url, {"text": "a bowl"})
    assert "no longer answers" in e.value.message and "preempted" in e.value.message
    assert "/health" in e.value.fix and "again" in e.value.fix


def test_a_slow_request_times_out_with_the_reason(serve, monkeypatch):
    monkeypatch.setattr(remote, "TIMEOUT_S", 0.3)
    with pytest.raises(CreateError) as e:
        remote.post_generate(serve(["slow"]), {"text": "a bowl"})
    assert "no answer" in e.value.message and "loads the models" in e.value.message and "busy" in e.value.fix


def test_health_says_token_busy_and_loaded():
    from alloy.create.serve import health

    h = health("secret")
    assert h["ok"] and h["token_required"] is True and h["busy"] is False and isinstance(h["loaded"], list)
    assert health("")["token_required"] is False
