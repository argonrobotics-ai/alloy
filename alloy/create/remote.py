# SPDX-License-Identifier: Apache-2.0
"""The client side of a generation server (``python -m alloy.create.serve``): one request, with an error that says
what went wrong and what to do for each way it can fail.

Before the request, ``GET /health`` (10 s) tells a server that is down from one that refuses the token. During
the request, a dropped connection is retried once if the server still answers ``/health`` (a restarted server),
and reported as the machine having gone away if it does not (a spot GPU machine can be preempted at any time).

Example::

    >>> data = post_generate("http://gpu-box:8765", {"text": "a ceramic bowl", "seed": 0})   # doctest: +SKIP
    >>> sorted(data)                                                                           # doctest: +SKIP
    ['image_png_b64', 'mesh_obj_b64', 'models', 'prompt', 'seconds']
"""

from __future__ import annotations

import http.client
import json
import logging
import os
import socket
import urllib.error
import urllib.request
from typing import Any

from alloy.create.request import CreateError

log = logging.getLogger("alloy.create")

URL_ENV = "ALLOY_GEN3D_URL"
TOKEN_ENV = "ALLOY_GEN3D_TOKEN"
TIMEOUT_S = 900                      # per socket read: the first request after a start loads the models (1-2 min)
HEALTH_TIMEOUT_S = 10
RETRIES = 1                          # resend once after a dropped connection when the server still answers
_DROPPED = (http.client.RemoteDisconnected, http.client.IncompleteRead, ConnectionResetError, BrokenPipeError)


def post_generate(url: str, body: dict[str, Any]) -> dict[str, Any]:
    """``POST <url>/generate`` with ``body``; the server's JSON reply. Raises ``CreateError`` (message + fix)."""
    base = url.rstrip("/")
    token = os.environ.get(TOKEN_ENV, "").strip()
    health = _preflight(base, bool(token))
    for attempt in range(RETRIES + 1):
        try:
            return _post(base + "/generate", body, token)
        except urllib.error.HTTPError as exc:
            raise _http_error(base, exc, bool(token)) from exc
        except Exception as exc:  # noqa: BLE001 - sorted below into dropped, timed out and unreachable
            if not _was_dropped(exc):
                raise _failure(base, exc, health) from exc
            if attempt == RETRIES or health_of(base) is None:
                raise _dropped(base, exc) from exc
            log.warning("create: the generator at %s dropped the connection; sending the request again", base)
    raise CreateError("no attempt was made", "report this; RETRIES must be >= 0", field="input")


def _was_dropped(exc: Exception) -> bool:
    """The connection closed mid-request (raw, or wrapped in a URLError when it happened while sending)."""
    return isinstance(exc, _DROPPED) or isinstance(getattr(exc, "reason", None), _DROPPED)


def _failure(base: str, exc: Exception, health: dict[str, Any]) -> CreateError:
    if isinstance(exc, (TimeoutError, socket.timeout)) or isinstance(getattr(exc, "reason", None), TimeoutError):
        return _timeout(base, health)
    if isinstance(exc, urllib.error.URLError):
        return _unreachable(base, exc.reason)
    return CreateError(f"the generator at {base} sent a reply this client cannot read ({type(exc).__name__}: "
                       f"{exc})", f"check that {URL_ENV} points at `python -m alloy.create.serve`", field="input")


def health_of(base: str) -> dict[str, Any] | None:
    """The server's ``GET /health`` reply, or None when nothing answers within ``HEALTH_TIMEOUT_S``."""
    try:
        with urllib.request.urlopen(base + "/health", timeout=HEALTH_TIMEOUT_S) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as exc:  # noqa: BLE001 - any failure means "not answering"; the caller says what to do
        log.debug("create: %s/health: %s", base, exc)
        return None


def _preflight(base: str, have_token: bool) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(base + "/health", timeout=HEALTH_TIMEOUT_S) as resp:
            health = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise _http_error(base, exc, have_token) from exc
    except (TimeoutError, socket.timeout, ValueError, *_DROPPED) as exc:
        raise _unreachable(base, exc) from exc
    except urllib.error.URLError as exc:
        raise _unreachable(base, exc.reason) from exc
    if health.get("token_required") and not have_token:
        raise CreateError(f"the generator at {base} needs a token and none is set",
                          f"set {TOKEN_ENV} to the server's token (ask whoever runs it)", field="input")
    return health


def _post(url: str, body: dict[str, Any], token: str) -> dict[str, Any]:
    req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), method="POST",
                                 headers={"Content-Type": "application/json"})
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _http_error(base: str, exc: urllib.error.HTTPError, have_token: bool) -> CreateError:
    if exc.code == 401:
        why = f"the token in {TOKEN_ENV} is not the server's" if have_token else "it needs a token and none was sent"
        return CreateError(f"the generator at {base} refused the request: {why}",
                           f"set {TOKEN_ENV} to the server's token (ask whoever runs it)", field="input")
    try:
        sent = json.loads(exc.read().decode("utf-8"))
    except Exception:  # noqa: BLE001 - not a JSON error body (a proxy, another service on that port)
        sent = {}
    return CreateError(f"the generator at {base} answered HTTP {exc.code}: {sent.get('error', exc.reason)}",
                       sent.get("fix") or f"check that {URL_ENV} points at `python -m alloy.create.serve`",
                       field="input")


def _unreachable(base: str, reason: Any) -> CreateError:
    if isinstance(reason, ConnectionRefusedError):
        return CreateError(f"nothing is listening at {base}: the generator server is not running there",
                           f"start it on the GPU machine (`python -m alloy.create.serve`), check {base}/health, and "
                           f"check the port in {URL_ENV}", field="input")
    return CreateError(f"cannot reach the generator at {base} ({type(reason).__name__}: {reason})",
                       f"check {URL_ENV}, the network, and that the GPU machine is up (a spot machine may have been "
                       f"stopped or preempted); {base}/health answers when it is ready", field="input")


def _dropped(base: str, exc: Exception) -> CreateError:
    if health_of(base) is None:
        return CreateError(f"the generator at {base} dropped the connection and no longer answers "
                           f"({type(exc).__name__}): its machine went away (a spot GPU machine can be preempted) or "
                           "the server stopped", f"restart the machine or the server, wait until {base}/health "
                           "answers, then run the same request again", field="input")
    return CreateError(f"the generator at {base} closed the connection twice without an answer "
                       f"({type(exc).__name__}) but still answers /health", "look at the server's log for a crash "
                       "(GPU memory, a bad image), then run the same request again", field="input")


def _timeout(base: str, health: dict[str, Any]) -> CreateError:
    cold = "" if health.get("loaded") else "; the first request after a start also loads the models (1-2 min)"
    return CreateError(f"no answer from the generator at {base} within {TIMEOUT_S} s{cold}",
                       f"it handles one request at a time: check {base}/health ('busy') and the server log, then run "
                       "the same request again", field="input")


__all__ = ["HEALTH_TIMEOUT_S", "RETRIES", "TIMEOUT_S", "TOKEN_ENV", "URL_ENV", "health_of", "post_generate"]
