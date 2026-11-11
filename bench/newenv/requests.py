# SPDX-License-Identifier: Apache-2.0
"""Load the environment requests (requests.json) and the engine arms (engines.json).

Example:
    >>> reqs = load_requests()
    >>> reqs[0].id, reqs[0].checker
    ('E01', 'push_cube')
    >>> load_engines()["alloy"].agent_s
    2700
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger("bench.newenv")
HERE = Path(__file__).resolve().parent
EXTRA_ENV = "NEWENV_EXTRA_CONFIG"   # a directory whose engines.json / allowlist.json are laid over the public ones
GROUPS = ("rigid", "articulated", "soft", "liquid", "robot")


class ConfigError(ValueError):
    """requests.json or engines.json is malformed; the message names the entry and the fix."""


@dataclass(frozen=True)
class Request:
    id: str
    group: str
    request: str
    checker: str
    probe: dict[str, list] = field(default_factory=dict)   # key -> trailing shape; a str entry ("V", "P", "K") is a count
    policy: bool = False                                    # robot task: env.py must provide scripted_policy
    checks: tuple[str, ...] = ()                            # hidden from agents
    entry: dict[str, Any] = field(default_factory=dict)     # the whole entry with the file's defaults merged in


@dataclass(frozen=True)
class EngineSpec:
    key: str
    name: str
    port: int
    module: str
    lane: str
    install: str
    docs: str
    agent_s: int
    rerun_s: int
    minutes: float
    refs: str = ""
    declines: tuple[str, ...] = ()      # groups / request ids where a decline is correct (docs say unsupported)
    notice: str = ""                    # non-empty: this arm's rows may not be published; outputs start with it


def _read(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"{path}: {exc}") from exc


def parse_request(d: dict[str, Any], defaults: dict[str, Any] | None = None) -> Request:
    missing = [k for k in ("id", "group", "request", "checker") if not d.get(k)]
    if missing:
        raise ConfigError(f"request {d.get('id', '?')}: missing {', '.join(missing)}")
    if d["group"] not in GROUPS:
        raise ConfigError(f"request {d['id']}: group {d['group']!r} not in {', '.join(GROUPS)}")
    probe = d.get("probe")
    if not isinstance(probe, dict) or not probe:
        raise ConfigError(f"request {d['id']}: probe must be a non-empty object of key -> trailing shape")
    return Request(d["id"], d["group"], d["request"], d["checker"], {k: list(v) for k, v in probe.items()},
                   bool(d.get("policy", False)), tuple(d.get("checks", ())), {**(defaults or {}), **d})


def load_requests(path: Path | None = None) -> list[Request]:
    raw = _read(path or HERE / "requests.json")
    reqs = [parse_request(d, raw.get("defaults")) for d in raw["requests"]]
    ids = [r.id for r in reqs]
    dup = sorted({i for i in ids if ids.count(i) > 1})
    if dup:
        raise ConfigError(f"duplicate request ids: {', '.join(dup)}")
    return reqs


def _overlay(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    """``extra`` laid over ``base``: an object value merges one level down (an arm's fields, the allowlist ports)."""
    out = dict(base)
    for key, val in extra.items():
        out[key] = {**out[key], **val} if isinstance(val, dict) and isinstance(out.get(key), dict) else val
    return out


def read_config(name: str, path: Path | None = None) -> dict[str, Any]:
    """``name`` (engines.json, allowlist.json) from this package, with $NEWENV_EXTRA_CONFIG/<name> laid over it.

    An explicit ``path`` is read alone. The extra directory holds arms that are not part of the public package (for
    example one whose results may not be published: its entry carries a ``notice``).
    """
    if path is not None:
        return _read(path)
    raw = _read(HERE / name)
    extra = os.environ.get(EXTRA_ENV)
    if extra and (Path(extra) / name).is_file():
        raw = _overlay(raw, _read(Path(extra) / name))
    return raw


def load_engines_raw(path: Path | None = None) -> dict[str, dict[str, Any]]:
    return {k: v for k, v in read_config("engines.json", path).items() if not k.startswith("_")}


def load_engines(path: Path | None = None) -> dict[str, EngineSpec]:
    raw = read_config("engines.json", path)
    defaults = raw.get("_defaults", {})
    out = {}
    for key, d in raw.items():
        if key.startswith("_"):
            continue
        g = {**defaults, **d}
        out[key] = EngineSpec(key, d["name"], int(d["port"]), d["module"], d.get("lane", "gpu"), d["install"],
                              d["docs"], int(g["agent_s"]), int(g["rerun_s"]), float(g["minutes"]),
                              d.get("refs", ""), tuple(d.get("declines", ())), d.get("notice", ""))
    return out


def is_correct_decline(engine: EngineSpec, req: Request) -> bool:
    """Do the engine's public docs say it cannot do this request (its group or id is in ``declines``)?"""
    return req.group in engine.declines or req.id in engine.declines


def unfinished_engines(keys: list[str], path: Path | None = None) -> dict[str, list[str]]:
    """Engines among ``keys`` whose entry still holds a TODO value -> the fields that do (blocks a launch)."""
    raw = load_engines_raw(path)
    bad = {k: [f for f, v in raw[k].items() if isinstance(v, str) and "TODO" in v] for k in keys if k in raw}
    return {k: v for k, v in bad.items() if v}


__all__ = ["ConfigError", "EXTRA_ENV", "EngineSpec", "GROUPS", "Request", "is_correct_decline", "load_engines",
           "load_engines_raw", "load_requests", "parse_request", "read_config", "unfinished_engines"]
