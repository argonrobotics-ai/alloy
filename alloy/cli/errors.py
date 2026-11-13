# SPDX-License-Identifier: Apache-2.0
"""Errors of the command line and tool server, and the JSON form of any error an operation raises.

Every error becomes ``{"type", "message", "fix", "field"}``: what went wrong and what to change. Example::

    from alloy.cli.errors import RequestError, error_payload
    error_payload(RequestError("step: steps must be integer", "pass steps=10", field="steps"))
    # {"type": "RequestError", "message": "step: steps must be integer", "fix": "pass steps=10", "field": "steps"}
"""

from __future__ import annotations

import logging
from typing import Any

from alloy.api.errors import AlloyAPIError, remedy_for
from alloy.core.errors import AlloyError

log = logging.getLogger("alloy.cli")

#: exception class name -> what to change, for errors from packages that carry no fix of their own
_FIXES: dict[str, str] = {
    "FileNotFoundError": "pass the path of an existing file (relative paths start in the current directory)",
    "PermissionError": "write to a folder you can write to (out=...)",
    "ImportFileError": "pass an MJCF (.xml), URDF (.urdf) or USD (.usd/.usda/.usdc) file; the message names the "
                       "problem",
    "ExportError": "the message names the object that did not export; export it alone or drop it",
    "SceneInputError": "pass an existing scene .yaml path or a library scene name (alloy list scenes)",
    "CameraError": "the message names the camera argument to change",
    "CardError": "change the card field named in the message",
}


class RequestError(AlloyAPIError, ValueError):
    """A request the tool cannot run as given (unknown operation or argument, wrong type, missing value).

    Example: ``RequestError("unknown op 'stpe'", "use one of: build, step, ...", field="op")``."""


class SessionError(AlloyAPIError, LookupError):
    """A simulation id that is not open, or too many open simulations. Example: ``step`` with ``sim="nope"``."""


#: longest message or fix a reply carries: an error echoing a huge input (a 1 MB id) stays small
MAX_TEXT = 4000


def bounded(text: Any, limit: int = MAX_TEXT) -> Any:
    """``text`` cut to ``limit`` characters with a note of how much was left out (non-strings unchanged).

    Example: ``len(bounded("x" * 10_000)) < 4100``."""
    if not isinstance(text, str) or len(text) <= limit:
        return text
    return f"{text[:limit]}... ({len(text) - limit} more characters)"


def error_payload(exc: BaseException) -> dict[str, Any]:
    """JSON form of ``exc``: ``{"type", "message", "fix", "field"}``, each text at most ``MAX_TEXT`` characters.
    API errors keep their own fix; engine errors get the API's remedy table; anything unexpected is logged with its
    traceback and says how to report it.

    Example: ``error_payload(FileNotFoundError("x.xml"))["fix"]``."""
    payload = _payload(exc)
    return {k: bounded(v, 200 if k == "field" else MAX_TEXT) for k, v in payload.items()}


def _payload(exc: BaseException) -> dict[str, Any]:
    if isinstance(exc, AlloyAPIError):
        d = exc.to_dict()
        return {"type": d["error"], "message": d["message"], "fix": d["fix"], "field": d["field"]}
    name = type(exc).__name__
    fix = next((f for cls in type(exc).__mro__ for k, f in _FIXES.items() if cls.__name__ == k), None)
    expected = fix is not None or isinstance(exc, (AlloyError, ValueError, LookupError, OSError))
    fix = fix or remedy_for(exc)
    if expected or fix != remedy_for(Exception()):
        return {"type": name, "message": str(exc), "fix": fix, "field": getattr(exc, "field", None)}
    log.error("unexpected %s", name, exc_info=exc)
    return {"type": name, "message": str(exc) or name, "field": None,
            "fix": "this looks like a bug in Alloy: report it with the request that caused it (the traceback is on "
                   "stderr)"}
