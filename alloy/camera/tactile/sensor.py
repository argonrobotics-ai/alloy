# SPDX-License-Identifier: Apache-2.0
"""``TactileSensor``: a GelSight / DIGIT-style sensor on a body, rendered for every world like ``alloy.camera.Camera``.

Example (a ball pressed 0.5 mm into a pad, CPU)::

    from alloy._vendor import newton; import warp as wp
    from alloy.camera.tactile import TactileSensor
    b = newton.ModelBuilder()
    pad = b.add_body(xform=wp.transform((0, 0, 0), wp.quat_identity()), label="pad")
    b.add_shape_box(pad, hx=0.012, hy=0.01, hz=0.002)
    ball = b.add_body(xform=wp.transform((0, 0, 0.0095), wp.quat_identity()), label="ball")
    b.add_shape_sphere(ball, radius=0.008)
    model = b.finalize(device="cpu")
    sensor = TactileSensor("pad_gel", attach_to="pad", pose=((0, 0, 0.002), (0, 0, 0, 1)), preset="gelsight",
                           width=160)
    img = sensor.render(model, model.state())          # wp.array uint8 (1, 120, 160, 3)
"""
from __future__ import annotations

import logging

import numpy as np
import warp as wp

from ..errors import CameraError
from .augment import Augment
from .depth import PressMap
from .markers import MarkerMotion
from .particles import ParticlePress
from .renderer import OUTPUTS, TactileOutput, TactileRenderer
from .shear import StickSlip

log = logging.getLogger(__name__)


class TactileSensor:
    """A vision-based tactile sensor over every world: its gel sits at ``pose`` in the ``attach_to`` body frame.

    ``pose``: the gel surface centre, +Z pointing out of the gel toward what it touches, +X / +Y the image's right /
    down; (px, py, pz, qx, qy, qz, qw), ``((px, py, pz), (qx, qy, qz, qw))`` or per-world (worlds, 7) [m].
    ``attach_to``: a body label (each world's copy is found), a body index, one body index per world, or None
    (the gel is fixed in the world). ``preset``: "gelsight" or "digit" (or a TactilePreset). ``width`` /
    ``height``: image size (the gel's aspect is kept; default 320 x 240). ``markers``: None (preset default), True,
    False or a MarkerSpec. ``augment``: an :class:`Augment` (e.g. ``Augment.strong(seed)``) for sim-to-real.
    ``ignore``: body labels or indices the gel does not feel (besides its own body). ``slip_mm`` / ``slip_rad``:
    shear and twist the gel holds before the object slips. ``deformables`` (default True): cloth and soft-body
    surfaces, MPM grains and liquid particles press the gel too (``alloy.camera.tactile.particles``); rigid shapes
    are traced exactly either way.
    """

    def __init__(self, name: str = "tactile", attach_to=None, pose=None, preset="gelsight", width: int | None = None,
                 height: int | None = None, markers=None, augment: Augment | None = None, ignore=(),
                 slip_mm: float = 0.6, slip_rad: float = 0.35, marker_motion: MarkerMotion | None = None,
                 deformables: bool = True):
        from .presets import load_preset
        self.name, self.attach_to, self.ignore = name, attach_to, tuple(ignore)
        self.deformables = bool(deformables)
        self.preset = load_preset(preset)
        self._opts = dict(width=width, height=height, markers=markers, augment=augment, motion=marker_motion)
        self._pose = _as_pose(pose)
        self.slip = (float(slip_mm), float(slip_rad))
        self._bound = None  # (model, PressMap, StickSlip, TactileRenderer, ParticlePress or None)

    # ------------------------------------------------------------------ public
    def render(self, source, state=None, output: str = "uint8", shear_field: wp.array | None = None,
               seed: int | None = None):
        """Images of every world at ``state``: ``source`` is an ``alloy.Sim`` or a newton.Model (+ its state).

        Returns a wp.array uint8 (worlds, H, W, 3) (row 0 = top of the sensor image), or for ``output="all"`` a
        :class:`TactileOutput` with the press map [mm] (the depth image in the sensor frame), the spread gel
        surface, marker centres and the stick-slip state. ``shear_field``: an engine tangential gel displacement
        field (worlds, H, W) vec2 [mm] that moves the markers instead of the stick-slip estimate. The arrays are
        reused by the next render: copy what you keep. Example: ``img = sensor.render(sim)``.
        """
        if output not in OUTPUTS:
            raise CameraError(f"output {output!r}", f"use one of {OUTPUTS}")
        model, state = _unwrap(source, state)
        pm, ss, tr, pp = self._bind(model, source)
        body_q = getattr(state, "body_q", None) if state is not None else None
        if body_q is None and pm.max_list > 0 and model.body_count > 0:
            raise CameraError(f"tactile sensor {self.name!r} got no state",
                              "pass the simulation state: sensor.render(model, state)")
        press = pm.compute(body_q)
        if pp is not None:
            pp.add(state, pm.gel_xf, press, pm.touched)
        tr.prepare(press, pm.touched)
        motion = None
        if shear_field is None and tr.grid is not None and body_q is not None:
            motion = ss.update(tr.optics.stats, pm.gel_xf, pm._shape_body, body_q)
        out = tr.finish(motion, ss.center, shear_field, output, seed)
        if output == "all":
            return TactileOutput(out.rgb, out.press, out.gel, out.markers, motion)
        return out

    def render_torch(self, source, state=None, **kw):
        """Same as :meth:`render` (uint8), as a torch tensor on the same device (needs PyTorch)."""
        out = self.render(source, state, **kw)
        if isinstance(out, TactileOutput):
            raise CameraError("render_torch with output='all'", "use render(...) and wp.to_torch on each field")
        try:
            return wp.to_torch(out)
        except ImportError as exc:
            raise CameraError("PyTorch is not installed", "pip install torch, or use render() for warp arrays") from exc

    def set_pose(self, pose) -> None:
        """New gel pose in the body frame (same forms as the constructor's ``pose``)."""
        self._pose = _as_pose(pose)
        if self._bound is not None:
            self._bound[1].set_local_pose(self._pose)

    def reset(self, worlds=None, seed: int | None = None) -> None:
        """Forget the ongoing touches of the worlds a reset restarted: ``worlds`` is a bool (worlds,) mask (numpy,
        torch or warp), a list of world indices, or None for every world. ``seed`` also redraws those worlds'
        augmentation. The other worlds keep their stick-slip state and augmentation.

        Example: ``env.reset_worlds(done); sensor.reset(worlds=done)``."""
        if self._bound is None:
            return
        mask = None if worlds is None else _world_mask(worlds, self._bound[2].motion.shape[0])
        dev = self._bound[2].device
        self._bound[2].reset(None if mask is None else wp.array(mask.astype(np.int32), dtype=wp.int32, device=dev))
        if seed is not None:
            self._bound[3].resample(seed, worlds=mask)

    @property
    def size(self) -> tuple[int, int]:
        """(height, width) of the images."""
        from .presets import output_size
        return output_size(self.preset, self._opts["width"], self._opts["height"])

    # ------------------------------------------------------------------ internals
    def _bind(self, model, source=None):
        if self._bound is not None and self._bound[0] is model:
            return self._bound[1:]
        names = getattr(source, "index", None)
        bodies = _sim_bodies(names, self.attach_to)
        if bodies is None:
            bodies = _resolve_bodies(model, self.attach_to)
        skip = tuple(_resolve_ignore(model, self.ignore, names))
        h, w = self.size
        n = len(bodies)
        opts = {k: v for k, v in self._opts.items() if k not in ("width", "height")}
        tr = TactileRenderer(self.preset, w, h, n, str(model.device), **opts)
        pm = PressMap(model, bodies, h, w, tr.mm_per_px, self.preset.max_depth_mm, self.preset.gel_radius_mm,
                      skip_bodies=skip)
        pm.set_local_pose(self._pose)
        ss = StickSlip(n, str(model.device), *self.slip)
        pp = None
        if self.deformables:
            pp = ParticlePress(model, n, h, w, tr.mm_per_px, self.preset.max_depth_mm, self.preset.gel_radius_mm,
                               sim=source if names is not None else None)
            pp = pp if pp.active else None
        self._bound = (model, pm, ss, tr, pp)
        log.info("tactile sensor %s: %d worlds, %dx%d, preset %s", self.name, n, w, h, self.preset.name)
        return pm, ss, tr, pp


def _world_mask(worlds, n: int) -> np.ndarray:
    """A bool (n,) mask from a mask (numpy / torch / warp) or a list of world indices."""
    if isinstance(worlds, wp.array):
        worlds = worlds.numpy()
    elif hasattr(worlds, "detach"):  # torch
        worlds = worlds.detach().cpu().numpy()
    a = np.asarray(worlds)
    if a.dtype == bool and a.shape == (n,):
        return a
    if a.dtype != bool and a.ndim <= 1 and (a.size == 0 or np.issubdtype(a.dtype, np.integer)):
        idx = a.astype(np.int64).reshape(-1)
        if idx.size and (idx.min() < 0 or idx.max() >= n):
            raise CameraError(f"worlds {idx.tolist()} outside 0..{n - 1}", f"pass world indices below {n}")
        out = np.zeros(n, bool)
        out[idx] = True
        return out
    raise CameraError(f"worlds has shape {a.shape} and dtype {a.dtype}",
                      f"pass a bool ({n},) mask or a list of world indices")


def _unwrap(source, state):
    engine = getattr(source, "engine", None)
    if engine is not None and hasattr(engine, "model"):
        return engine.model, (state if state is not None else getattr(engine, "state", None))
    if not hasattr(source, "shape_type"):
        raise CameraError(f"cannot sense a {type(source).__name__}", "pass an alloy.Sim or a newton.Model and state")
    return source, state


def _as_pose(pose) -> np.ndarray:
    if pose is None:
        return np.array([0, 0, 0, 0, 0, 0, 1.0])
    if isinstance(pose, (tuple, list)) and len(pose) == 2 and np.size(pose[0]) == 3:
        return np.concatenate([np.asarray(pose[0], np.float64), np.asarray(pose[1], np.float64)])
    return np.asarray(pose, np.float64)


def _labels(model) -> list[str]:
    return list(getattr(model, "body_label", None) or getattr(model, "body_key", None) or [])


def _num_worlds(model) -> int:
    return max(1, int(getattr(model, "world_count", 0) or getattr(model, "num_worlds", 0) or 1))


def _resolve_bodies(model, attach_to) -> np.ndarray:
    """One body index per world for ``attach_to`` (-1 everywhere when None)."""
    n = _num_worlds(model)
    if attach_to is None:
        return np.full(n, -1, np.int32)
    if np.ndim(attach_to) == 1:
        out = np.asarray(attach_to, np.int32)
        if out.shape != (n,):
            raise CameraError(f"attach_to has {out.size} bodies for {n} worlds", "give one body index per world")
        return out
    labels = _labels(model)
    if isinstance(attach_to, (int, np.integer)):
        if not 0 <= int(attach_to) < model.body_count:
            raise CameraError(f"attach_to={attach_to} is not a body index", f"use 0..{model.body_count - 1}")
        if n == 1 or not labels:
            return np.full(n, int(attach_to), np.int32)
        attach_to = labels[int(attach_to)]
    return _per_world(model, labels, str(attach_to), n)


def _per_world(model, labels: list[str], name: str, n: int) -> np.ndarray:
    hits = [i for i, lb in enumerate(labels) if lb == name or lb.endswith("/" + name)]
    if not hits:
        raise CameraError(f"no body {name!r}", f"use one of the body labels, e.g. {labels[:4]}")
    world = model.body_world.numpy() if getattr(model, "body_world", None) is not None else np.zeros(len(labels))
    out = np.full(n, -1, np.int32)
    for i in hits:
        w = int(world[i])
        if 0 <= w < n and out[w] < 0:
            out[w] = i
    shared = [i for i in hits if int(world[i]) < 0]
    if shared:
        out[out < 0] = shared[0]
    if (out < 0).any():
        raise CameraError(f"no body {name!r} in world {int(np.flatnonzero(out < 0)[0])}",
                          "attach the sensor to a body that every world has")
    return out


def _sim_bodies(names, item) -> np.ndarray | None:
    """Per-world body indices of an ``alloy.Sim`` name ("cube", "gripper/finger_left"), else None."""
    if names is None or not isinstance(item, str) or item not in getattr(names, "bodies", {}):
        return None
    return names.per_world(names.body(item), "body").astype(np.int32)


def _resolve_ignore(model, ignore, names=None) -> list[int]:
    labels, out = _labels(model), []
    for item in ignore:
        if isinstance(item, (int, np.integer)):
            out.append(int(item))
            continue
        sim_hits = _sim_bodies(names, item)
        if sim_hits is not None:
            out.extend(int(b) for b in sim_hits)
            continue
        hits = [i for i, lb in enumerate(labels) if lb == item or lb.endswith("/" + str(item))]
        if not hits:
            raise CameraError(f"ignore={item!r} matches no body", f"use body labels, e.g. {labels[:4]}")
        out.extend(hits)
    return out
