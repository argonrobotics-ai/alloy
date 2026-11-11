# SPDX-License-Identifier: Apache-2.0
"""``Tactile``: taxel arrays on robot links from the solver's own contact forces (engine item 10, tactile v1).

Each pad is a body label, plus a pose in that body's frame (``origins``, ``rotations`` as x, y, z, w). The pad's z is
its outward normal; its x and y span the taxel layout. Every step, each contact on a pad body within
``capture_depth`` of the pad plane adds its force to the nearest taxel within ``reach``, or shares it by a Gaussian
of width ``spread``, in the style of MuJoCo's touch_grid. Channels per taxel, in N:
- 0: normal force, compression positive;
- 1 and 2: shear along the pad's x and y.

The noise layer (``noise.py``) then applies. Readings (``read()["<label>"]``):
- ``taxels`` [W, P, T, 3];
- ``pad_force`` [W, P, 3], the noise-free total over every captured contact, the reference for the force-sum law.

Needs contact forces (``needs_contacts``): the MuJoCo route on MuJoCo Warp, where the plug-in host exports the
solver's contacts and CONTACT_F forces (about +1-3 % per step).

Example::

    t = Tactile(bodies=("fr3_leftfinger", "fr3_rightfinger"), origins=((0, 0.008, 0.045),) * 2,
                rotations=((0.7071, 0, 0, 0.7071),) * 2, preset="xela_uskin_4x4", noise=0.01)
    sim = alloy.plugins.simulation(built, [t]); sim.step_period(); sim.plugins.read()["tactile"]["taxels"]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
import warp as wp

from alloy._vendor import newton

from alloy.plugins.base import SensorPlugin
from alloy.plugins.errors import PluginError
from alloy.plugins.tactile import kernels as K
from alloy.plugins.tactile import noise as N
from alloy.plugins.tactile.layout import Layout, build_layout


class Tactile(SensorPlugin):
    """Taxel force arrays (normal + 2 shear) on pads of named bodies, binned from contact forces, with a noise
    layer and real-sensor presets. See the module docstring for frames and channels."""

    name = "tactile"
    summary = "taxel arrays (normal + 2 shear per taxel) on link pads, from contact forces; noise layer; presets"
    needs_contacts: ClassVar[bool] = True

    @dataclass
    class Config:
        bodies: tuple[str, ...] = ("pad",)                                     # one pad per body label
        origins: tuple[tuple[float, float, float], ...] = ((0.0, 0.0, 0.0),)  # pad centre, body frame (m)
        rotations: tuple[tuple[float, float, float, float], ...] = ((0.0, 0.0, 0.0, 1.0),)  # pad frame (x,y,z,w)
        grid: tuple[int, int] = (4, 4)                                         # nx, ny taxels
        pitch: tuple[float, float] = (0.004, 0.004)                            # m between taxel centres
        foveation: float = 0.0                                                 # > 0 packs taxels toward the centre
        points: tuple[tuple[float, float], ...] = ()                           # m, taxel centres instead of a grid
        reach: float = 0.0                       # m: max contact-to-taxel distance; 0 = from the layout
        capture_depth: float = 0.003             # m: contacts within ±this of the pad plane count
        spread: float = 0.0                      # m: Gaussian share width; 0 = nearest taxel
        preset: str = ""                         # a presets.PRESETS name: sets the layout and saturation
        noise: float = 0.0                       # N, white noise std
        drift: float = 0.0                       # N/sqrt(s), bias random walk
        tau_load: float = 0.0                    # s, hysteresis lag while the force rises
        tau_unload: float = 0.0                  # s, while it falls
        dead: float = 0.0                        # fraction of dead taxels per world
        crosstalk: float = 0.0                   # share of each 4-neighbour's signal (grids)
        saturation: float = 0.0                  # N, clip per channel; 0 = none
        lsb: float = 0.0                         # N, ADC step; 0 = none
        seed: int = 0

        def __post_init__(self) -> None:
            if not self.bodies:
                raise ValueError("bodies: at least one pad body label")
            for name in ("origins", "rotations"):
                if len(getattr(self, name)) not in (1, len(self.bodies)):
                    raise ValueError(f"{name}: one entry, or one per body ({len(self.bodies)})")
            if self.capture_depth <= 0.0 or self.spread < 0.0 or self.reach < 0.0:
                raise ValueError("capture_depth must be > 0; spread and reach >= 0")
            N.NoiseParams(*self._noise_tuple())  # validates the noise settings

        def _noise_tuple(self) -> tuple:
            return (self.noise, self.drift, self.tau_load, self.tau_unload, self.dead, self.crosstalk,
                    self.saturation, self.lsb, self.seed)

    def setup(self, ctx: Any) -> None:
        c, W = self.config, ctx.world_count
        self.layout: Layout = build_layout(c)
        if self.layout.neighbours is None and c.crosstalk > 0.0:
            raise PluginError("tactile: crosstalk needs a grid layout (points have no 4-neighbours)")
        self.P, self.T = len(c.bodies), len(self.layout.xy)
        self._resolve_pads(ctx)
        shape = (W, self.P, self.T, 3)
        self.taxels = self.add_state("taxels", ctx.zeros(shape))
        self.pad_force = self.add_state("pad_force", ctx.zeros((W, self.P, 3)))
        self.hyst = self.add_state("hysteresis", ctx.zeros(shape))
        self.bias = self.add_state("bias", ctx.zeros(shape))
        self.tick = self.add_state("tick", ctx.zeros(W, wp.int32))
        slots = W * self.P * self.T
        self._acc = ctx.zeros(slots * 3, wp.int64)
        self._pad_acc = ctx.zeros(W * self.P * 3, wp.int64)
        self._raw = ctx.zeros(slots * 3)
        self._dead = ctx.array(N.dead_mask(W, self.P * self.T, c.dead, c.seed), wp.int32)
        nb = self.layout.neighbours if self.layout.neighbours is not None else -np.ones(self.T * 4, np.int32)
        self._neighbours = ctx.array(nb, wp.int32)
        self._flat = {k: a.reshape((-1,)) for k, a in (("taxels", self.taxels), ("pad_force", self.pad_force),
                                                        ("hyst", self.hyst), ("bias", self.bias))}

    def _resolve_pads(self, ctx: Any) -> None:
        c, model = self.config, ctx.model
        owner = np.full(model.body_count, -1, np.int32)
        for p, label in enumerate(c.bodies):
            for w, b in enumerate(ctx.bodies(label).numpy()):
                if owner[b] != -1:
                    raise PluginError(f"tactile: body {label!r} carries two pads; one pad per body per sensor "
                                      f"(use a second Tactile with its own label)")
                owner[b] = w * self.P + p
        self._pad_of_body = ctx.array(owner, wp.int32)
        origins = [c.origins[min(p, len(c.origins) - 1)] for p in range(self.P)]
        rots = np.asarray([c.rotations[min(p, len(c.rotations) - 1)] for p in range(self.P)], np.float64)
        rots /= np.linalg.norm(rots, axis=1, keepdims=True)
        self._origin = ctx.array(np.asarray(origins, np.float32), wp.vec3)
        self._rot = ctx.array(rots.astype(np.float32), wp.quat)
        self._xy = ctx.array(self.layout.xy.astype(np.float32), wp.vec2)

    def measure(self, state_in: newton.State, state_out: newton.State, dt: float) -> None:
        ctx, c = self.ctx, self.config
        con, force = ctx.contacts, ctx.contact_force
        if con is None or force is None:
            raise PluginError("tactile needs contact forces: run it on the MuJoCo route (MuJoCo Warp), where the "
                              "plug-in host exports them")
        model, dev, slots = ctx.model, ctx.device, ctx.world_count * self.P * self.T
        wp.launch(K.bin_contacts, dim=con.rigid_contact_max, device=dev,
                  inputs=[con.rigid_contact_count, con.rigid_contact_shape0, con.rigid_contact_shape1,
                          con.rigid_contact_point0, con.rigid_contact_point1, force, model.shape_body,
                          state_in.body_q, self._pad_of_body, self.P, self._origin, self._rot, self._xy, self.T,
                          float(self.layout.reach), float(c.capture_depth), float(c.spread)],
                  outputs=[self._acc, self._pad_acc])
        wp.launch(K.read_out, dim=slots, device=dev, inputs=[self._acc, self._pad_acc, self.T],
                  outputs=[self._raw, self._flat["pad_force"]])
        wp.launch(N.apply_noise, dim=slots, device=dev,
                  inputs=[self._raw, self._neighbours, self.T, self.P, self.tick, self._dead,
                          wp.vec4(c.noise, c.drift, c.tau_load, c.tau_unload),
                          wp.vec4(c.crosstalk, self.layout.saturation, c.lsb, 0.0), int(c.seed), float(dt),
                          self._flat["hyst"], self._flat["bias"]],
                  outputs=[self._flat["taxels"]])
        wp.launch(N.advance_tick, dim=ctx.world_count, device=dev, inputs=[self.tick])

    def test_scene(self):
        from alloy.plugins.tactile.laws import rest_scene  # noqa: PLC0415 - laws import this module

        return rest_scene()

    def laws(self):
        from alloy.plugins.tactile.laws import tactile_laws  # noqa: PLC0415

        return tactile_laws()
