# SPDX-License-Identifier: Apache-2.0
"""``BondField``: the ``Bonds`` plug-in's part inside the film solver.

``attach`` wraps that solver instance's ``step``, as ``PluginHost`` wraps MuJoCo's for the contact export, and adds a
VBD colour callback. Each film step then runs:
1. pin bonded vertices for the strain projection (inverse mass 0) and let released grips go;
2. the film step, with the implicit bond springs in VBD's ``after_accumulate`` slot;
3. formation from the step's contacts, then the wet-out update (``wetout.py``);
4. the peel front (``front.py``).
Every launch has a fixed size, so the step can be captured as a graph. A coupler's iterations re-run the step from the
same input; the bond state is restored before each re-run (``before``), so each exchange updates it once.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import warp as wp

from . import front as F
from . import kernels as K

log = logging.getLogger("alloy.plugins.bonds")


class BondField:
    """Film-solver-side state and launches. ``arrays`` are the plug-in's flat views (``plugin.Bonds._flat``),
    ``init`` the host-side layout (``plugin.Bonds._layout``)."""

    def __init__(self, solver: Any, arrays: dict[str, wp.array], init: dict[str, Any], *, et: float,
                 form: bool) -> None:
        self.solver, self.a, self.et, self.form = solver, arrays, float(et), bool(form)
        m, dev = solver.model, solver.model.device
        self.model, self.n, self.dev = m, int(m.particle_count), dev
        self.world = m.particle_world
        arr = lambda v, t: wp.array(np.ascontiguousarray(v), dtype=t, device=dev)  # noqa: E731
        self.inv_mass0 = arr(m.particle_inv_mass.numpy(), float)   # read once: every writer on main runs at build
        self.k = arr(init["k"], float)
        self.cell = arr(init["cell"], float)
        self.key = wp.zeros(self.n, dtype=wp.int64, device=dev)
        self.fresh = wp.zeros(self.n, dtype=wp.int32, device=dev)
        self.rest_dir = arr(init["rest_dir"], wp.vec3)
        self.surface = K.Surface()
        for name in ("substrate", "p_form", "dwell", "ke", "margin", "tau_w"):
            setattr(self.surface, name, arr(init["surface"][name], wp.int32 if name == "substrate" else float))
        self.peel = F.Peel()
        for name in ("g90", "g180", "tau"):
            setattr(self.peel, name, arr(init["peel"][name], float))
        self.grip_mass = arr(init["grip_mass"], float)
        self.press = K.Press()
        self.press.rollers, self.press.world = init["rollers"], m.particle_world
        self.press.shape_xform = m.shape_transform
        self.bonded = K.Bonded()
        self.bonded.state, self.bonded.shape = arrays["bond_state"], arrays["bond_shape"]
        self.bonded.world, self.bonded.base = m.particle_world, arr(init["base"], wp.int32)
        self.bonded.anchor, self.bonded.k = arrays["anchor"], self.k
        self.film = self._film_struct(init, arr)
        self.front = self._front_struct()
        self.no_body = wp.zeros(1, dtype=wp.transform, device=dev)
        self.mutable = {k: v for k, v in arrays.items() if k not in ("release_grip", "roller_load")}
        self.shadow = {k: wp.clone(v) for k, v in self.mutable.items()}
        self.exchange_dt = 0.0                        # set by Bonds.apply, once per exchange
        self.calls = 0                                # film steps so far in this exchange
        self.any_substrate = bool(np.any(init["surface"]["substrate"]))
        self.dt = 0.0
        self.air = init.get("air")                    # AirField (air.py) with Config.trapped_air, else None

    def _film_struct(self, init: dict[str, Any], arr: Any) -> F.Film:
        fm = F.Film()
        fm.start = arr(init["start"], wp.int32)
        fm.area, fm.h, fm.d = arr(init["area"], float), arr(init["h"], float), arr(init["d"], float)
        fm.gnear = arr(init["gnear"], wp.int32)
        fm.state, fm.shape, fm.base = self.a["bond_state"], self.a["bond_shape"], self.bonded.base
        fm.normal, fm.bond_dir, fm.wet = self.a["normal"], self.a["bond_dir"], self.a["wet"]
        fm.q = self.model.particle_q
        return fm

    def _front_struct(self) -> F.Front:
        fr = F.Front()
        for name in ("a", "xg_prev", "stretch", "force", "point", "front_point", "grip_shape", "sub_shape",
                     "peel_force",
                     "angle", "width", "bonded", "released", "dissipated", "work", "impulse", "front"):
            setattr(fr, name, self.a[name])
        return fr

    # --- install ----------------------------------------------------------------------------------------------
    def install(self) -> None:
        s = self.solver
        inner = s.step

        def step(state_in, state_out, control, contacts, dt, *args, **kwargs):
            self.before(state_in, dt)
            inner(state_in, state_out, control, contacts, dt, *args, **kwargs)
            self.after(state_out, contacts, dt)

        s.step = step
        s.add_colour_callback(self.accumulate, external=True, when="after_accumulate")
        if self.air is not None:
            s.add_colour_callback(self.air_accumulate, external=True, when="after_accumulate")
        log.info("bonds: attached to %s (%d particles)", type(s).__name__, self.n)

    def _body_q(self, state: Any) -> wp.array:
        bq = getattr(state, "body_q", None)
        return bq if bq is not None and bq.shape[0] > 0 else self.no_body

    # --- per step ---------------------------------------------------------------------------------------------
    def accumulate(self, solver: Any, colour: int, state: Any, contacts: Any) -> None:
        wp.launch(K.bond_accumulate, dim=self.n,
                  inputs=[colour, self.model.particle_colors, state.particle_q, self._body_q(state),
                          self.model.shape_body, self.bonded],
                  outputs=[solver.particle_forces, solver.particle_hessians], device=self.dev)

    def air_accumulate(self, solver: Any, colour: int, state: Any, contacts: Any) -> None:
        self.air.accumulate(solver, colour, state, contacts, self._body_q(state))

    def before(self, state_in: Any, dt: float) -> None:
        """A coupler may step the film several times per exchange: k substeps of dt = exchange/k, each sequence possibly
        repeated as coupling iterations from the same input. The first substep of the first iteration saves the bond
        state; the first substep of a later iteration restores it; other substeps advance. So each exchange's update
        counts once, and substeps keep theirs (DC5 review S1)."""
        k = max(1, int(round(self.exchange_dt / dt))) if self.exchange_dt > 0.0 else 1
        c, self.calls = self.calls, self.calls + 1
        if c % k == 0:
            for name, v in self.mutable.items():
                if c == 0:
                    wp.copy(self.shadow[name], v)
                else:
                    wp.copy(v, self.shadow[name])
        wp.launch(K.release_grips, dim=self.n, inputs=[self.a["release_grip"], self.world, self.a["bond_state"]],
                  device=self.dev)
        wp.launch(K.pin_sync, dim=self.n, inputs=[self.a["bond_state"], self.inv_mass0],
                  outputs=[self.model.particle_inv_mass], device=self.dev)

    def after(self, state_out: Any, contacts: Any, dt: float) -> None:
        bq = self._body_q(state_out)
        if self.form and self.any_substrate and contacts is not None and contacts.soft_contact_max > 0:
            self._formation(state_out, contacts, bq, dt)
        self.film.q = state_out.particle_q
        wp.launch(F.peel_front, dim=self.front.a.shape[0],
                  inputs=[self.film, self.peel, self.front, bq, self.model.shape_body, self.et, self.grip_mass,
                          float(dt), float(self.exchange_dt if self.exchange_dt > 0.0 else dt)], device=self.dev)
        if self.air is not None:
            self.air.after(state_out, bq, dt)

    def _formation(self, state_out: Any, contacts: Any, bq: wp.array, dt: float) -> None:
        ct = K.Contacts()
        ct.count, ct.particle, ct.shape = (contacts.soft_contact_count, contacts.soft_contact_particle,
                                           contacts.soft_contact_shape)
        ct.body_pos, ct.normal = contacts.soft_contact_body_pos, contacts.soft_contact_normal
        q, r, sb, a = state_out.particle_q, self.model.particle_radius, self.model.shape_body, self.a
        self.key.zero_()
        dim = int(contacts.soft_contact_max)
        rl = self.press.rollers
        if rl.count > 0:                              # each roller's line load, measured on the film (wetout.py)
            wp.launch(K.roller_force, dim=dim, inputs=[ct, self.surface, q, r, bq, sb, self.press], device=self.dev)
            wp.launch(K.roller_load, dim=rl.load.shape, inputs=[rl, float(dt)], device=self.dev)
        wp.launch(K.press_keys, dim=dim, inputs=[ct, self.surface, q, r, bq, sb, self.cell, self.press],
                  outputs=[self.key], device=self.dev)
        if self.air is not None:
            self.air.block(self.key)
        wp.launch(K.press_dwell, dim=self.n, inputs=[self.key, self.surface, float(dt), a["bond_state"],
                                                     a["dwell"], self.fresh], device=self.dev)
        wp.launch(K.form_bonds, dim=dim,
                  inputs=[ct, self.surface, q, r, bq, sb, self.cell, self.press, self.key, self.fresh, self.rest_dir,
                          self.bonded, a["normal"], a["bond_dir"], a["dwell"]],
                  device=self.dev)
        wp.launch(K.wet_update, dim=self.n, inputs=[self.key, self.surface, self.press, self.bonded, q, r, a["normal"],
                                                    bq, sb, float(dt)], outputs=[a["wet"]], device=self.dev)


__all__ = ["BondField"]
