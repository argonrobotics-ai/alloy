# SPDX-License-Identifier: Apache-2.0
"""RenderRecorder: capture Alloy/Newton physics frames into an ARB v1 bundle (see docs/RENDER.md)."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from .bundle import BundleError
from .extract import to_np
from .plan import Decls, Layout, build_layout
from .writer import write_bundle

log = logging.getLogger("alloy.render")


class RenderRecorder:
    """Records exported worlds of a Newton model each ``capture`` and writes the bundle in ``close``.

    Declare ``add_*`` sets before the first ``capture``; the layout is frozen then.
    """

    def __init__(self, model, out_dir, fps: float, worlds="all", grid_m: float | None = None,
                 meta: dict | None = None, leftover_points: bool = True) -> None:
        if not fps or fps <= 0:
            raise BundleError(f"fps must be positive, got {fps!r}")
        self.model, self.root, self.fps = model, Path(out_dir), float(fps)
        self.worlds, self.grid_m, self.meta = worlds, grid_m, dict(meta or {})
        self._decls = Decls(leftover_points=bool(leftover_points))
        self._layout: Layout | None = None
        self._t: list[float] = []
        self._bq: list[np.ndarray] = []
        self._pq: list[np.ndarray] = []
        self._pqd: list[np.ndarray] = []
        self._closed = False

    def add_points(self, name: str, particle_ids, radius: float | None = None) -> None:
        """Name a particle set (SPH liquid, MPM grains, ...)."""
        self._declare(self._decls.points, (name, particle_ids, radius))

    def add_surface(self, name: str, tri, kind: str = "cloth") -> None:
        """Declare a triangle surface from ``tri`` (F, 3) model particle ids."""
        if kind not in ("cloth", "soft"):
            raise BundleError(f"surface {name!r}: kind must be 'cloth' or 'soft', got {kind!r}")
        self._declare(self._decls.surfaces, (name, tri, kind))

    def add_curve_from_particles(self, name: str, ids, radius: float | None = None) -> None:
        """A particle rod, in order, rendered as one smooth tube."""
        self._declare(self._decls.curve_particles, (name, ids, radius))

    def add_curve_from_bodies(self, name: str, regex_or_ids, radius: float | None = None) -> None:
        """A chain of capsule/cylinder bodies rendered as one tube along their centreline."""
        self._declare(self._decls.curve_bodies, (name, regex_or_ids, radius))

    def add_instancer(self, name: str, regex_or_body_ids) -> None:
        """Many one-shape bodies with identical geometry, exported as one instanced object."""
        self._declare(self._decls.instancers, (name, regex_or_body_ids))

    def skip_particles(self, particle_ids) -> None:
        """Never auto-export these particles (e.g. the MPM particles a grain instancer does not draw)."""
        self._declare(self._decls.skip, ("skip_particles", particle_ids))

    def _declare(self, target: list, item: tuple) -> None:
        if self._closed or self._layout is not None:
            raise BundleError(f"cannot declare {item[0]!r} after the first capture or close")
        target.append(item)

    def _ensure_layout(self) -> Layout:
        if self._layout is None:
            self._layout = build_layout(self.model, self._decls, self.worlds, self.grid_m)
            log.info("recorder layout: %d bodies, %d shapes, %d surfaces, %d point sets, %d curves, %d instancers",
                     len(self._layout.bodies), len(self._layout.shapes), len(self._layout.surfaces),
                     len(self._layout.points), len(self._layout.curves), len(self._layout.instancers))
        return self._layout

    def capture(self, state, t: float) -> None:
        """Copy the exported subset of ``state`` to host RAM at time ``t``."""
        if self._closed:
            raise BundleError("capture() after close()")
        lay = self._ensure_layout()
        t = float(t)
        k = len(self._t)
        if not np.isfinite(t) or (k and t <= self._t[-1]):
            prev = self._t[-1] if k else None
            raise BundleError(f"frame {k}: t={t} must be finite and greater than previous t={prev}")
        bq = self._gather("body_q", state.body_q, lay.cap_bodies, 7, k)
        pq = self._gather("particle_q", state.particle_q, lay.cap_particles, 3, k)
        pqd = self._gather("particle_qd", state.particle_qd, lay.cap_particles, 3, k) if lay.need_velocity else None
        self._t.append(t)
        self._bq.append(bq)
        self._pq.append(pq)
        if pqd is not None:
            self._pqd.append(pqd)

    @staticmethod
    def _gather(name: str, arr, rows: np.ndarray, width: int, frame: int) -> np.ndarray:
        if len(rows) == 0:
            return np.zeros((0, width), np.float32)
        if arr is None:
            raise BundleError(f"frame {frame}: state.{name} is None but the export needs it")
        out = np.ascontiguousarray(to_np(arr)[rows], dtype=np.float32)
        if not np.isfinite(out).all():
            bad = int(np.argwhere(~np.isfinite(out))[0][0])
            raise BundleError(f"frame {frame}: state.{name} has NaN/Inf (first at exported row {bad}, "
                              f"source index {int(rows[bad])})")
        return out

    def close(self) -> Path:
        """Stack the captures and write the bundle; returns its directory."""
        if self._closed:
            raise BundleError("close() called twice")
        if not self._t:
            raise BundleError("close() with no captured frames")
        lay = self._ensure_layout()
        self._closed = True
        device = str(getattr(self.model, "device", "unknown"))
        world_count = int(getattr(self.model, "world_count", 0))
        write_bundle(self.root, lay, self.fps, self.meta, device, world_count, self._t, self._bq, self._pq, self._pqd)
        self._bq, self._pq, self._pqd = [], [], []
        return self.root
