# SPDX-License-Identifier: Apache-2.0
"""``GelSensor``: the physics side of vision-based tactile sensors (GelSight / DIGIT style; tactile v2).

The split with OSS-4 and the interface spec are in section 6 of [internal note]. This
plug-in writes the fields the image side renders, on the last solver step of each control period (``every``).
``outputs()`` holds, in the sensor frame:
- ``depth`` [W, S, Hg, Wg]: gel indentation in m, positive into the gel; row 0 at +y;
- ``marker_disp`` [W, S, M, 3]: x, y tangential, z out of the gel;
- ``force`` [W, S, 3]: the engine's normal force and its two shear forces on the pad, in N;
- ``in_contact`` [W, S] int32;
- ``stamp`` [W] int32 steps.

The sensor frame:
- its origin is the gel centre, with z out of the gel, toward the object;
- the in-sensor camera looks along +z: image x = sensor x, image y up = sensor y;
- grid cell (i, j) covers image pixels [i·s, (i+1)·s) x [j·s, (j+1)·s).

How the fields are computed (``gel_kernels.py``, ``gel_loads.py``):
- **depth** is the engine's own penetration of the touching shapes into the undeformed gel surface, sphere-traced on
  Newton's public SDFs. So the scene gives the pad a gel-like contact compliance; a stiff pad gives faint images.
- **marker_disp** superposes a Green's tensor over loads in proportion to depth. Those loads carry the engine's
  normal force, shear force and twist exactly. The scale is set so the normal displacement at the peak equals the
  depth there.

Example::

    gel = GelSensor(bodies=("left_pad",), origins=((0, 0, 0.004),), preset="gelsight", markers=(7, 9))
    sim = alloy.plugins.simulation(built, [gel]); sim.step_period(); d = gel.outputs()["depth"].numpy()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
import warp as wp

from alloy._vendor import newton

from alloy.plugins.base import SensorPlugin
from alloy.plugins.errors import PluginError
from alloy.plugins.tactile import gel_kernels as GK
from alloy.plugins.tactile import gel_loads as GL
from alloy.plugins.tactile.noise import advance_tick

GEL_PRESETS = {   # image (H, W) px and m per px from Taxim's Basics/sensorParams.py (commit 4936657a)
    "gelsight": {"image": (480, 640), "pixel": 0.0295e-3},   # Taxim's original GelSight calibration (calibs/ root)
    "digit": {"image": (480, 640), "pixel": 0.0266e-3},      # a comment there; confirmed on Taxim's DIGIT data pack
}                                                            # by OSS-21's scale scan (2 mm ball = 75.2 px)


class GelSensor(SensorPlugin):
    """Gel penetration depth field and marker displacement field per sensor, from the engine's contacts (see the
    module docstring for frames, units and method)."""

    name = "gel_tactile"
    summary = "vision-tactile physics: gel penetration depth field + marker displacements from contacts (v2)"
    needs_contacts: ClassVar[bool] = True

    @dataclass
    class Config:
        bodies: tuple[str, ...] = ("pad",)                                     # one gel per body label
        origins: tuple[tuple[float, float, float], ...] = ((0.0, 0.0, 0.0),)  # gel centre, body frame (m)
        rotations: tuple[tuple[float, float, float, float], ...] = ((0.0, 0.0, 0.0, 1.0),)  # gel frame (x,y,z,w)
        preset: str = "gelsight"                 # sets image and pixel; "" = use them as given
        image: tuple[int, int] = (480, 640)      # px, H x W
        pixel: float = 0.0295e-3                 # m per px
        stride: int = 4                          # px per depth cell
        block: int = 4                           # depth cells per coarse load cell (side)
        max_depth: float = 0.002                 # m, gel thickness cap
        capture_depth: float = 0.005             # m: contacts within ±this of the gel plane count
        markers: tuple[int, int] = (0, 0)        # rows, cols of a uniform marker grid (no source gives a layout)
        marker_margin: float = 0.1               # fraction of the field of view left free at each edge
        kernel: str = "analytic"                 # "analytic" (half-space) or "taxim" (FEM tensor, fetched)
        poisson: float = 0.48                    # gel Poisson ratio (analytic kernel)
        every: int = 0                           # compute the fields every N solver steps; 0 = once per period

        def __post_init__(self) -> None:
            for name in ("origins", "rotations"):
                if len(getattr(self, name)) not in (1, len(self.bodies)):
                    raise ValueError(f"{name}: one entry, or one per body ({len(self.bodies)})")
            if self.preset and self.preset not in GEL_PRESETS:
                raise ValueError(f"no gel preset {self.preset!r}; presets: {sorted(GEL_PRESETS)}")
            if self.kernel not in ("analytic", "taxim"):
                raise ValueError("kernel is 'analytic' or 'taxim'")
            if min(self.stride, self.block) < 1 or self.max_depth <= 0.0 or not 0.0 <= self.poisson < 0.5 \
                    or self.every < 0:
                raise ValueError("stride, block >= 1; max_depth > 0; 0 <= poisson < 0.5; every >= 0")

    def setup(self, ctx: Any) -> None:
        c, W = self.config, ctx.world_count
        self._geometry(c)
        self.S = len(c.bodies)
        G = W * self.S
        self._resolve_pads(ctx)
        self._acc = {"force": ctx.zeros(G * 3, wp.int64), "torque": ctx.zeros(G, wp.int64)}
        self._slots = ctx.full(G * GK.SLOTS, -1, wp.int32)
        self._depth = ctx.zeros(G * self.rows * self.cols)
        self._load = ctx.zeros(G * self.cr * self.cc)
        self._prm = ctx.zeros(G * GL.P_PARAMS)
        self.force = self.add_state("force", ctx.zeros((W, self.S, 3)))
        self.in_contact = self.add_state("in_contact", ctx.zeros((W, self.S), wp.int32))
        self.stamp = self.add_state("stamp", ctx.zeros(W, wp.int32))
        M = max(self.marker_xy.shape[0], 1)
        self._marker_xy = ctx.array(self.marker_xy.astype(np.float32) if len(self.marker_xy) else
                                    np.zeros((1, 2), np.float32), wp.vec2)
        self.marker_disp = self.add_state("marker_disp", ctx.zeros((W, self.S, M), wp.vec3))
        self._table = self._kernel_table(ctx)
        self._flat = {"force": self.force.reshape((-1,)), "contact": self.in_contact.reshape((-1,)),
                      "markers": self.marker_disp.reshape((-1,))}

    def _geometry(self, c: Any) -> None:
        image, pixel = (GEL_PRESETS[c.preset]["image"], GEL_PRESETS[c.preset]["pixel"]) if c.preset else \
            (c.image, c.pixel)
        h, w = int(image[0]), int(image[1])
        if h % c.stride or w % c.stride or (h // c.stride) % c.block or (w // c.stride) % c.block:
            raise PluginError(f"gel_tactile: image {h}x{w} must divide into stride {c.stride} x block {c.block}")
        self.pixel, self.rows, self.cols = float(pixel), h // c.stride, w // c.stride
        self.cell, self.cr, self.cc = c.stride * self.pixel, self.rows // c.block, self.cols // c.block
        self.coarse = self.cell * c.block
        self.marker_xy = marker_grid(c.markers, (w * self.pixel, h * self.pixel), c.marker_margin)

    def _resolve_pads(self, ctx: Any) -> None:
        c = self.config
        owner = np.full(ctx.model.body_count, -1, np.int32)
        pad_body = np.zeros(ctx.world_count * self.S, np.int32)
        for p, label in enumerate(c.bodies):
            for w, b in enumerate(ctx.bodies(label).numpy()):
                if owner[b] != -1:
                    raise PluginError(f"gel_tactile: body {label!r} carries two gels; one per body per sensor")
                owner[b] = w * self.S + p
                pad_body[w * self.S + p] = b
        rots = np.asarray([c.rotations[min(p, len(c.rotations) - 1)] for p in range(self.S)], np.float64)
        rots /= np.linalg.norm(rots, axis=1, keepdims=True)
        self._pad_of_body, self._pad_body = ctx.array(owner, wp.int32), ctx.array(pad_body, wp.int32)
        self._origin = ctx.array(np.asarray([c.origins[min(p, len(c.origins) - 1)] for p in range(self.S)],
                                            np.float32), wp.vec3)
        self._rot = ctx.array(rots.astype(np.float32), wp.quat)

    def _kernel_table(self, ctx: Any) -> wp.array:
        if self.config.kernel != "taxim":
            return ctx.array(np.zeros((1, 3, 3), np.float32), wp.mat33)
        from alloy.plugins.tactile.taxim import fem_tensor_sensor_axes  # noqa: PLC0415 - fetches on demand

        return ctx.array(fem_tensor_sensor_axes().reshape(-1, 3, 3), wp.mat33)

    def _green_args(self) -> list:
        from alloy.plugins.tactile.taxim import FEM_PIXMM  # noqa: PLC0415

        return [float(self.coarse), float(self.config.poisson), int(self.config.kernel == "taxim"), self._table,
                float(FEM_PIXMM)]

    def measure(self, state_in: newton.State, state_out: newton.State, dt: float) -> None:
        ctx, c, m = self.ctx, self.config, self.ctx.model
        con, frc = ctx.contacts, ctx.contact_force
        if con is None or frc is None:
            raise PluginError("gel_tactile needs contact forces: run it on the MuJoCo route (MuJoCo Warp), where the "
                              "plug-in host exports them")
        dev, G = ctx.device, ctx.world_count * self.S
        n = int(c.every) or max(1, int(round(ctx.control_dt / dt)))      # the last step of each period by default
        gate = [self.S, self.stamp, n]
        self._slots.fill_(-1)
        wp.launch(GK.gel_contacts, dim=con.rigid_contact_max, device=dev,
                  inputs=[con.rigid_contact_count, con.rigid_contact_shape0, con.rigid_contact_shape1,
                          con.rigid_contact_point0, con.rigid_contact_point1, frc, m.shape_body, state_in.body_q,
                          self._pad_of_body, self.S, self._origin, self._rot, float(c.capture_depth), self.stamp, n],
                  outputs=[self._acc["force"], self._acc["torque"], self._slots])
        wp.launch(GK.gel_depth, dim=G * self.rows * self.cols, device=dev,
                  inputs=[self._slots, self._pad_body, self.S, self._origin, self._rot, self.rows, self.cols,
                          float(self.cell), float(c.max_depth), state_in.body_q, m.shape_body, m.shape_transform,
                          m.shape_type, m.shape_scale, m.shape_source_ptr, self.stamp, n], outputs=[self._depth])
        wp.launch(GL.gel_coarse, dim=G * self.cr * self.cc, device=dev,
                  inputs=[self._depth, self.rows, self.cols, int(c.block), *gate], outputs=[self._load])
        flat = self._flat
        wp.launch(GL.gel_sensor, dim=G, device=dev,
                  inputs=[self._acc["force"], self._acc["torque"], self._load, self.cr, self.cc, float(self.coarse),
                          int(c.block * c.block), *self._green_args(), *gate, self._prm],
                  outputs=[flat["force"], flat["contact"]])
        if len(self.marker_xy):
            wp.launch(GL.gel_markers, dim=G * len(self.marker_xy), device=dev,
                      inputs=[self._marker_xy, len(self.marker_xy), self._load, self.cr, self.cc, float(self.coarse),
                              *self._green_args(), self._prm, flat["contact"], *gate],
                      outputs=[flat["markers"]])
        wp.launch(advance_tick, dim=ctx.world_count, device=dev, inputs=[self.stamp])

    def outputs(self) -> dict[str, wp.array]:
        """The device arrays the image side reads after each step (shapes in the module docstring)."""
        W = self.ctx.world_count
        return {"depth": self._depth.reshape((W, self.S, self.rows, self.cols)), "marker_disp": self.marker_disp,
                "force": self.force, "in_contact": self.in_contact, "stamp": self.stamp}

    def test_scene(self):
        from alloy.plugins.tactile.gel_laws import press_scene  # noqa: PLC0415 - laws import this module

        return press_scene()

    def laws(self):
        from alloy.plugins.tactile.gel_laws import gel_laws  # noqa: PLC0415

        return gel_laws()


def marker_grid(shape: tuple[int, int], fov: tuple[float, float], margin: float) -> np.ndarray:
    """[rows·cols, 2] marker rest positions (m, sensor frame), uniform over the field of view minus ``margin``."""
    rows, cols = int(shape[0]), int(shape[1])
    if rows < 1 or cols < 1:
        return np.zeros((0, 2))
    w, h = fov[0] * (1.0 - 2.0 * margin), fov[1] * (1.0 - 2.0 * margin)
    xs = np.linspace(-w / 2, w / 2, cols) if cols > 1 else np.zeros(1)
    ys = np.linspace(h / 2, -h / 2, rows) if rows > 1 else np.zeros(1)
    return np.stack(np.meshgrid(xs, ys), -1).reshape(-1, 2)
