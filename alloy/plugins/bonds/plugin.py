# SPDX-License-Identifier: Apache-2.0
"""``Bonds``: cohesive bonds between a film (SolverFilm, route ``cloth_film``) and rigid surfaces. They form under
pressure and peel at a resolved front, with Kendall's energy balance and a G(theta) calibrated from tape data.

**Bonded vertices.** Each film vertex has one bond slot: free, substrate (bonded to a surface) or grip (held by the
body that holds the tab).
- A bond is an implicit spring k = ``stiffness``·A_i to an anchor fixed in the shape's body frame, in VBD's colour
  callback. It holds statically to nanometres.
- A bonded vertex is also pinned for SolverFilm's strain projection (inverse mass 0; VBD keeps its mass). The free arm
  keeps the film's inextensibility, gravity and contact.

**Formation.** A free vertex in contact with a substrate shape bonds when two conditions hold:
- its elastic contact pressure p = k_c·penetration/a_cell is at least ``p_form``. k_c is the contact stiffness VBD uses
  and a_cell the film's area per vertex. VBD gives every vertex the same contact stiffness, so a uniform press loads
  every vertex alike and p reads the nominal F/A. Under a roller (``rollers``), p is its rubber cover's Hertz line
  contact instead, which does not depend on the mesh (wetout.py);
- the pressure has lasted for ``dwell`` without a break.
It is anchored at the surface point, offset by the contact radius.

**Wet-out (wetout.py).** Each vertex's wetted fraction w grows while pressed at p >= p_form, dw/dt = (1 - w)·(p/P_REF)^M
/tau_w, with tau_w calibrated so a standard application (the ASTM D3330 roller, twice) gives W_STD = 0.90. A bond's G is
G_card·w/W_STD, so the cards' rows hold for a standard application; pre-bonded starts begin at W_STD.

**Peel (front.py).**
- The front sits at material distance a from the grip vertices, and bonds with rest distance d_i < a are released.
- The arm is a straight line from the grip to the front. Its tension T = min(E·t·b·s/a, T_c) comes from its stretch s,
  where T_c = b·E·t·(sqrt(c² + 2G(c)/(E·t)) - c) is Kendall with the elastic term, and c = 1 - cos(theta).
- The front advances only as far as keeps T <= T_c, then:
  - a slack arm (s <= 0) pulls with 0;
  - a robot weaker than T_c stalls the front.
- -T·u acts on the grip body at the grip point, and +T·u on a dynamic substrate at the front. Both are applied one
  exchange late, through ``body_f``.
- On a dynamic grip the arm stiffness is capped at 0.02·m_grip/dt², with critical damping 2·sqrt(k·m_grip) on its
  stretch rate (standing in for the backing's and adhesive's viscoelasticity). That keeps the lagged, explicit pull
  stable (front.py ARM_CAP). In steady peeling the force is T_c either way; the front then lags the inextensible arm
  by T_c/k (2.4 mm at 9.5 N on a 0.2 kg grip at 1 ms).
- The film's own internal forces on the grip vertices are not passed to the grip body: the arm's weight is < 1e-3
  of T for tapes and labels.

**Limits (v1):**
- Quasi-static: no rate dependence. Kendall's balance gives the front speed, so inertia in the arm is ignored.
- Lap shear is capped at ``tau``·(peelable area).
- The cohesive zone is not resolved: it is below the mesh scale, so the front is a line.
- G(theta) is a calibration through both 7818 rows (adhesives.py).
- One grip per world. The front lies across the peel direction, at one material distance from the grip.
- Peel geometry comes from the film's rest layout: build films flat. A film may start posed (a label wrapped on a
  tube): its material distances then come from the unfolded rest mesh (layout.unfold), its directions from the pose.
- Grips are boxes; pre-bonded substrates are boxes, cylinders or capsules (formation works on any shape).
- Wet-out grows with pressure and time in one rate law (wetout.py ESTIMATES); its 10 min to 72 h growth after the
  press is not modelled.
- Formation still needs the card's dwell above p_form (0.05 s, ESTIMATE). Under the standard roller a vertex stays
  above p_form for about 2.5 mm of travel, so bonds form only below about 50 mm/s; a faster applicator needs a
  card with a shorter dwell (w already carries the time dependence).
- A roller's Hertz contact uses 1/R_e = 1/R + 1/R_s on a cylinder or capsule substrate within 10° of parallel to it
  (a label rolled onto a tube along its axis), else R_e = R: a roller across a tube is a point contact, not modelled.
- A roller's load is what the film carries: give its shape no rigid collision with the substrate, or set
  ``roller_load``.
- Trapped air (``trapped_air``, air.py) is off by default. Each pocket is one pressure, venting through the throat
  on the label's edge; the 3 mm² barcode criterion is not scored at the cell sizes label stock needs.
- Film to film bonds (tape on tape, gift wrap) are not modelled.
- The bond callback is registered ``external=True``: SolverVBDMomentum's coarse momentum step is off for the whole
  film entry, so free film pieces lose momentum as without it. Bonded vertices get no air force and no inertia
  prediction (inverse mass 0).
- The film entry must own every particle of the model.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import warp as wp

from alloy._vendor import newton

from alloy.plugins.base import Forces, ForcePlugin
from alloy.plugins.errors import PluginError

from .adhesives import adhesive
from .air import AIR_ROUGHNESS, AIR_VISCOSITY, SLOT_STATE, AirField
from .air import READINGS as AIR_READINGS
from .air_kernels import SLOTS
from .field import BondField
from .kernels import apply_pull
from .wetout import W_STD, Roller, rollers_struct, wet_tau
from .wiring import film_solver, layout, per_world

log = logging.getLogger("alloy.plugins.bonds")

PARTICLE = (("bond_state", wp.int32), ("bond_shape", wp.int32), ("anchor", wp.vec3), ("normal", wp.vec3),
            ("bond_dir", wp.vec3), ("dwell", float), ("wet", float))
WORLD = (("a", float), ("xg_prev", wp.vec3), ("stretch", float), ("force", wp.vec3), ("point", wp.vec3),
         ("front_point", wp.vec3),
         ("grip_shape", wp.int32), ("sub_shape", wp.int32))
READINGS = ("peel_force", "angle", "width", "bonded", "released", "dissipated", "work", "impulse", "front")


class Bonds(ForcePlugin):
    """Film-to-rigid cohesive bonds (module docstring). Readings, per world:
    - ``peel_force`` N, ``angle`` deg, ``width`` m: the current pull, peel angle and front width;
    - ``bonded`` and ``released`` m²: the area bonded now, and the area peeled so far;
    - ``dissipated`` J: Σ G(theta)·A over released bonds;
    - ``work`` J: ∫ T·(u·dx_grip), the pull's work on the grip (the energy ledger: work = dissipated + arm terms);
    - ``impulse`` N·s: ∫ T dt; its rate is the time-averaged pull;
    - ``front`` m: the front's distance from the grip.
    Per vertex: ``bond_state`` (0 free, 1 substrate, 2 grip), ``bond_shape`` and ``wet`` (the wetted fraction w).
    Per world and roller: ``roller_q`` N/m, the line load in use (measured on the film). Inputs: ``release_grip`` [W],
    1 lets a world's grip bonds go (a label pad releasing); ``roller_load`` [W, R] N/m, > 0 replaces a roller's
    measured line load. With ``trapped_air``: per vertex ``air_slot``; per world and pocket slot ``air_amount`` Pa·m³,
    ``air_p`` Pa (the felt excess), ``air_area`` m², ``air_volume`` m³, ``air_enclosed``; per world ``trapped_area``
    m² (the largest enclosed pocket holding air), ``pockets``, ``air_overflow``, ``air_unsettled`` and the gas books
    ``air_formed``, ``air_vented``, ``air_vanished``, ``air_untracked`` Pa·m³ (air.py)."""

    name = "bonds"
    summary = "film-to-rigid bonds: form above a pressure, peel at a resolved front (Kendall, G(theta) from 3M 7818)"

    @dataclass
    class Config:
        substrates: tuple[str, ...] = ("plate",)      # shape labels bonds form on (or start bonded to)
        adhesives: tuple[str, ...] = ("3m_7818/stainless_steel",)   # one card per substrate (adhesives.py)
        grip: str = ""                                # shape label of the body holding the tab (a box)
        grip_reach: float = 1.0e-3                    # m: film vertices this close to the grip box are held
        prebond: bool = False                         # start bonded where the film touches a substrate box
        prebond_reach: float = 1.0e-3                 # m: ... within this gap beyond the contact radius
        stiffness: float = 1.0e10                     # N/m per m² of vertex area (implicit; ~nm static error)
        face_et: float = 3.2e5                        # N/m: E·t of the face stock (7818: 0.08 mm PET, E 4 GPa est.)
        form: bool = True                             # bonds form under pressure
        rollers: tuple[Roller, ...] = ()              # roller cards (wetout.Roller; their shapes in the scene)
        roller_reach: float = 2.0e-3                  # m beyond a roller's radius a vertex counts as under it
        trapped_air: bool = False                     # pockets of air under the film (air.py)
        air_viscosity: float = AIR_VISCOSITY          # Pa·s, the vents' laminar slot law
        air_roughness: float = AIR_ROUGHNESS          # m, the gas layer at contact (air.py)

        def __post_init__(self) -> None:
            self.substrates, self.adhesives = tuple(self.substrates), tuple(self.adhesives)
            self.rollers = tuple(r if isinstance(r, Roller) else Roller(**r) for r in self.rollers)
            if len(self.adhesives) != len(self.substrates):
                raise ValueError(f"one adhesive card per substrate: {len(self.substrates)} substrates, "
                                 f"{len(self.adhesives)} cards")
            for name in self.adhesives:
                adhesive(name)
            if min(self.grip_reach, self.prebond_reach) < 0.0 or min(self.stiffness, self.face_et,
                                                                       self.air_viscosity, self.air_roughness) <= 0.0:
                raise ValueError("grip_reach, prebond_reach >= 0; stiffness, face_et, air_viscosity, air_roughness > 0")

    def setup(self, ctx: Any) -> None:
        m, W = ctx.model, ctx.world_count
        n = int(m.particle_count)
        if n == 0 or n % W:
            raise PluginError(f"bonds: needs a film with the same particle count in every world ({n} particles, "
                              f"{W} worlds)")
        P = n // W
        self._flat: dict[str, wp.array] = {}
        for name, t in PARTICLE:
            self._flat[name] = self.add_state(name, ctx.zeros((W, P), t)).reshape((n,))
        for name, t in WORLD:
            self._flat[name] = self.add_state(name, ctx.zeros(W, t))
        for name in READINGS:
            self._flat[name] = self.add_state(name, ctx.zeros(W, wp.float64))
        self._flat["release_grip"] = self.add_input("release_grip", ctx.zeros(W, wp.int32))
        rollers = max(len(self.config.rollers), 1)
        self._flat["roller_load"] = self.add_input("roller_load", ctx.zeros((W, rollers), float))
        self._flat["roller_q"] = self.add_state("roller_q", ctx.zeros((W, rollers), float))
        if self.config.trapped_air:
            self._air_state(ctx, W, P)
        self.field: BondField | None = None

    def _air_state(self, ctx: Any, W: int, P: int) -> None:
        """Pocket state (air.py): slots per vertex and per world, readings; slots and roots start idle (-1)."""
        self._flat["air_slot"] = self.add_state("air_slot", ctx.zeros((W, P), wp.int32)).reshape((W * P,))
        self._flat["air_slot"].fill_(-1)
        for name, t in SLOT_STATE:
            self._flat[name] = self.add_state(name, ctx.zeros((W, SLOTS), t))
        self._flat["air_root"].fill_(-1)
        for name in AIR_READINGS:
            self._flat[name] = self.add_state(name, ctx.zeros(W, wp.float64))

    def attach(self, stepper: Any) -> None:
        """Called by ``PluginHost.install``: find the film solver and install the bonds in its step."""
        solver = film_solver(stepper, int(self.ctx.model.particle_count))
        init = layout(self, solver)
        self._wet_init(init)
        if self.config.trapped_air:
            subs = np.stack([per_world(self.ctx, label) for label in self.config.substrates], axis=1)
            init["air"] = AirField(solver.model, self._flat, init, subs, self.config.air_viscosity,
                                   self.config.air_roughness)
        self.field = BondField(solver, self._flat, init, et=self.config.face_et, form=self.config.form)
        self.field.install()

    def _wet_init(self, init: dict[str, Any]) -> None:
        """Wet-out set-up beside ``layout``: tau_w per substrate shape, pre-bonds at W_STD, the rollers' shapes."""
        sf, m = init["surface"], self.ctx.model
        sf["tau_w"] = np.array([wet_tau(p) if sub else 1.0 for p, sub in zip(sf["p_form"], sf["substrate"])])
        state = self._flat["bond_state"].numpy()
        self._flat["wet"].assign(np.where(state == 1, W_STD, 0.0).astype(np.float32))
        cards = self.config.rollers
        shapes = np.zeros((self.ctx.world_count, max(len(cards), 1)), np.int64)
        scale = m.shape_scale.numpy()
        for k, card in enumerate(cards):
            ids = np.asarray(self.ctx.shapes(card.shape).numpy(), np.int64)
            if np.any(ids < 0) or not np.allclose(scale[ids, 0], card.radius, rtol=1e-3):
                radii = scale[ids.clip(0), 0].tolist()
                raise PluginError(f"bonds: roller {card.shape!r} must be a capsule or cylinder of radius {card.radius} "
                                  f"m in every world (found ids {ids.tolist()}, radii {radii})")
            shapes[:, k] = ids
        curved = np.isin(m.shape_type.numpy(), (int(newton.GeoType.CYLINDER), int(newton.GeoType.CAPSULE)))
        sub_radius = np.where(curved & (sf["substrate"] == 1), scale[:, 0], 0.0)      # tubes under a roller: R_e
        init["rollers"] = rollers_struct(cards, shapes, int(m.shape_count), self._flat["roller_q"],
                                         self._flat["roller_load"], self.config.roller_reach, m.device, sub_radius)

    def apply(self, state: Any, forces: Forces, dt: float) -> None:
        if self.field is None:
            raise PluginError("bonds: not attached to a film solver (install the PluginHost on the stepper)")
        self.field.exchange_dt, self.field.calls = float(dt), 0
        m = self.ctx.model
        if state.body_q is None or m.body_count == 0:
            return
        f = self._flat
        wp.launch(apply_pull, dim=self.ctx.world_count,
                  inputs=[f["force"], f["point"], f["front_point"], f["grip_shape"], f["sub_shape"],
                          self.field.bonded.base, m.shape_body, state.body_q, m.body_com], outputs=[forces.body_f],
                  device=m.device)

    def test_scene(self) -> Any:
        """A short 90° peel on the coupled rig (two worlds) for the harness's contract and hook checks."""
        from .laws import scene  # noqa: PLC0415 - laws import this module
        from .rig import PeelRig  # noqa: PLC0415

        return scene(PeelRig(), "follow", 10, worlds=2)

    def laws(self) -> list:
        from .laws import bonds_laws  # noqa: PLC0415

        return bonds_laws()


__all__ = ["Bonds"]
