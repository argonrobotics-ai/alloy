# SPDX-License-Identifier: Apache-2.0
"""The viewer backends: which are installed, and how to make one (a vendored Newton viewer) from plain options.

Standard library only at import time: viser, pyglet, imgui_bundle, rerun, pxr, Warp and Newton load inside
``create`` when a backend is made. Example::

    from alloy.viewer._backends import BackendOptions, backends, create
    backends()["viser"]                       # {"installed": True, "install": 'pip install "viser>=1.1.1"', ...}
    viewer = create("null", BackendOptions(frames=10))
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import logging
import os
import re
import sys
from unittest import mock
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, fields
from typing import Any

from alloy.api.errors import FeatureUnavailableError, SceneError

log = logging.getLogger("alloy.viewer")


@dataclass(frozen=True)
class BackendSpec:
    """One backend: the import names it needs, the pip line that installs them, and what it shows."""

    name: str
    modules: tuple[str, ...]
    install: str
    about: str
    min_version: tuple[str, tuple[int, ...]] | None = None   # (distribution, oldest working version)


BACKENDS: dict[str, BackendSpec] = {
    "viser": BackendSpec("viser", ("viser",), 'pip install "viser>=1.1.1"',
                         "browser viewer: a local web server, open view.url", ("viser", (1, 1, 1))),
    "gl": BackendSpec("gl", ("pyglet", "imgui_bundle"), "pip install pyglet imgui_bundle",
                      "desktop OpenGL window (needs a display)"),
    "rerun": BackendSpec("rerun", ("rerun",), "pip install rerun-sdk",
                         "records the run to an .rrd file (path=, default alloy_run.rrd); port= also serves the web viewer"),
    "usd": BackendSpec("usd", ("pxr",), "pip install usd-core", "writes the run to a USD file (path=, frames=)"),
    "null": BackendSpec("null", (), "", "draws nothing (tests and timing)"),
}


@dataclass
class BackendOptions:
    """Options for a backend. Unknown keywords go to the Newton viewer's constructor unchanged (``extra``).

    ``port``: viser's port, or rerun's web port; ``path``: usd output file, or a rerun ``.rrd`` recording;
    ``frames``: frame budget of the null, usd and headless gl backends; ``fps``: usd time samples per second."""

    port: int | None = None
    host: str | None = None
    path: str | None = None
    frames: int | None = None
    fps: int | None = None
    width: int | None = None
    height: int | None = None
    headless: bool | None = None
    verbose: bool | None = None
    paused: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_kwargs(cls, options: Mapping[str, Any], *, paused: bool = False) -> BackendOptions:
        """Split ``launch(**options)`` into the known fields and the pass-through rest."""
        known = {f.name for f in fields(cls)} - {"extra", "paused"}
        picked = {k: v for k, v in options.items() if k in known}
        extra = {k: v for k, v in options.items() if k not in known}
        return cls(**picked, paused=bool(paused), extra=extra)


# --- what is installed ------------------------------------------------------------------------------------------


def _module_present(name: str) -> bool:
    """True when ``name`` can be imported (checked without importing it)."""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _dist_version(dist: str) -> str | None:
    """The installed version string of a distribution, or None."""
    try:
        return importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
        return None


def _version_tuple(text: str) -> tuple[int, ...]:
    """Leading numeric parts of a version: ``"1.1.1rc2"`` -> ``(1, 1, 1)``."""
    match = re.match(r"\d+(?:\.\d+)*", text.strip())
    return tuple(int(p) for p in match.group(0).split(".")) if match else ()


def problem(spec: BackendSpec) -> str | None:
    """Why ``spec`` cannot run here (a missing or too old package), or None when it can."""
    missing = [m for m in spec.modules if not _module_present(m)]
    if missing:
        return f"{' and '.join(missing)} {'is' if len(missing) == 1 else 'are'} not installed"
    if spec.min_version is None:
        return None
    dist, oldest = spec.min_version
    found = _dist_version(dist)
    if found is not None and _version_tuple(found) < oldest:
        return f"{dist} {found} is too old (the viewer needs {'.'.join(map(str, oldest))} or newer)"
    return None


def backends() -> dict[str, dict[str, Any]]:
    """Every backend: installed here or not, the pip line that installs it, and what it shows.

    Example: ``{"viser": {"installed": False, "install": 'pip install "viser>=1.1.1"', "about": ...}, ...}``."""
    return {name: {"installed": problem(spec) is None, "install": spec.install, "about": spec.about}
            for name, spec in BACKENDS.items()}


def spec_for(name: str) -> BackendSpec:
    """The spec of a backend name; an unknown name raises ``SceneError`` listing the valid ones."""
    spec = BACKENDS.get(str(name).lower()) if isinstance(name, str) else None
    if spec is None:
        raise SceneError(f"unknown viewer backend {name!r}", f"pass backend= one of {', '.join(BACKENDS)}",
                         field="backend")
    return spec


def require(spec: BackendSpec) -> None:
    """Raise ``FeatureUnavailableError`` (with the pip line) when ``spec``'s packages are missing or too old."""
    reason = problem(spec)
    if reason is None:
        return
    raise FeatureUnavailableError(f"the {spec.name} viewer backend cannot start: {reason}",
                                  f"{spec.install}, or pick an installed backend (alloy.viewer.backends())",
                                  field="backend")


# --- making one ---------------------------------------------------------------------------------------------------


def _drop_none(values: Mapping[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in values.items() if v is not None}


def _null(nv: Any, o: BackendOptions) -> tuple[Callable[..., Any], dict[str, Any]]:
    return nv.ViewerNull, _drop_none({"num_frames": o.frames})


def _viser(nv: Any, o: BackendOptions) -> tuple[Callable[..., Any], dict[str, Any]]:
    return nv.ViewerViser, _drop_none({"host": o.host, "port": o.port, "paused": o.paused, "verbose": o.verbose})


def _gl(nv: Any, o: BackendOptions) -> tuple[Callable[..., Any], dict[str, Any]]:
    kwargs = {"width": o.width, "height": o.height, "headless": o.headless, "paused": o.paused,
              "num_frames": o.frames}
    import pyglet  # noqa: PLC0415 - optional package; the option must be set before pyglet opens a display

    pyglet.options["headless"] = bool(o.headless)      # headless=True draws off screen (EGL), no display needed
    return nv.ViewerGL, _drop_none(kwargs)


RERUN_DEFAULT_PATH = "alloy_run.rrd"


def _rerun(nv: Any, o: BackendOptions) -> tuple[Callable[..., Any], dict[str, Any]]:
    """Safe by default: no server, no window; the run is recorded to ``path`` (``alloy_run.rrd``). ``port=`` serves
    Rerun's web viewer on that port as well (opt in)."""
    serve = o.port is not None
    kwargs = _drop_none({"web_port": o.port, "record_to_rrd": o.path or RERUN_DEFAULT_PATH})
    kwargs["serve_web_viewer"] = serve
    return _RerunFactory(nv.ViewerRerun), kwargs


class _RerunFactory:
    """Builds a ViewerRerun; with ``serve_web_viewer=False`` it keeps Rerun from spawning a native viewer process
    (which needs a display) so the run goes to the recording file only."""

    def __init__(self, cls: Callable[..., Any]) -> None:
        self.cls = cls

    def __call__(self, **kwargs: Any) -> Any:
        if kwargs.get("serve_web_viewer", True) or kwargs.get("address") is not None:
            return self.cls(**kwargs)
        import rerun  # noqa: PLC0415

        with mock.patch.object(rerun, "spawn", lambda *args, **kw: None):
            return self.cls(**kwargs)


def _usd(nv: Any, o: BackendOptions) -> tuple[Callable[..., Any], dict[str, Any]]:
    if not o.path:
        raise SceneError("the usd viewer backend needs an output file", "pass path='run.usd'", field="path")
    return nv.ViewerUSD, _drop_none({"output_path": str(o.path), "num_frames": o.frames, "fps": o.fps})


_FACTORIES: dict[str, Callable[[Any, BackendOptions], tuple[Callable[..., Any], dict[str, Any]]]] = {
    "null": _null, "viser": _viser, "gl": _gl, "rerun": _rerun, "usd": _usd,
}


def _is_display_error(exc: BaseException, kwargs: Mapping[str, Any]) -> bool:
    """True for a window that could not open: pyglet says so, or no display is set and off-screen was not asked."""
    if type(exc).__name__ == "NoSuchDisplayException" or "display" in str(exc).lower():
        return True
    no_display = not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    return no_display and sys.platform.startswith("linux") and not kwargs.get("headless")


def _construct(spec: BackendSpec, factory: Callable[..., Any], kwargs: dict[str, Any]) -> Any:
    """Call the Newton viewer's constructor, turning its install and display failures into API errors."""
    try:
        return factory(**kwargs)
    except ImportError as exc:
        raise FeatureUnavailableError(f"the {spec.name} viewer backend failed to import a package: {exc}",
                                      spec.install or "reinstall Alloy", field="backend") from exc
    except TypeError as exc:
        if "unexpected keyword argument" not in str(exc):
            raise
        raise SceneError(f"the {spec.name} viewer backend does not take that option: {exc}",
                         "pass port=, host=, path=, frames= or a keyword of the backend's Newton viewer",
                         field="options") from exc
    except Exception as exc:
        if spec.name != "gl" or not _is_display_error(exc, kwargs):
            raise
        raise FeatureUnavailableError(f"the gl viewer backend needs a display and found none ({exc})",
                                      "run it in a desktop session (DISPLAY set), pass headless=True to draw off "
                                      "screen, or use backend='viser' and open view.url in a browser",
                                      field="backend") from exc


def create(name: str, options: BackendOptions) -> Any:
    """A Newton viewer for backend ``name``, made on the current Warp device. Example: ``create("null", opts)``."""
    spec = spec_for(name)
    require(spec)
    from alloy._vendor.newton import viewer as nv  # noqa: PLC0415 - Newton and Warp load only when a viewer is made

    factory, kwargs = _FACTORIES[spec.name](nv, options)
    kwargs.update(options.extra)
    viewer = _construct(spec, factory, kwargs)
    log.info("viewer backend %s started", spec.name)
    return viewer
