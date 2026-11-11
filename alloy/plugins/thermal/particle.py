# SPDX-License-Identifier: Apache-2.0
"""``ParticleThermal``: temperature per particle on SPH and MPM particles (heat v2, docs/notes/heat-v2-design.md).

A force plug-in that adds no forces (any route; no contact export needed). Each exchange, after the solver step
(``update``):

1. neighbour rows over the thermal particles (``alloy.solvers.sph.neighbors.NeighborLists``: grouped by world,
   sorted by particle id);
2. per particle, dH/dt from Cleary-Monaghan conduction (energy-conserving per pair; harmonic-mean conductivity across
   materials), heaters (a box shape at a fixed temperature through ``heater_conductance``, or a fixed power spread over
   the touching particles by area) and optional convection on exposed surfaces (``particle_kernels`` doc: Areas);
3. explicit Euler on the specific enthalpy H, then T(H) and the liquid fraction φ (phase_materials module doc);
4. melt -> flow (``feedback="auto"``): a flowing material writes μ(T, φ) into ``model.sph.viscosity`` and μ, τ_y(φ),
   friction(φ), E(φ) into ``model.mpm.*`` for its particles (``phase_kernels``), whichever arrays the model has.
   ``apply`` writes them again from the plug-in's state before every solver step, so after ``sim.reset()`` or a
   rollback (which restore T but not these model arrays) the solver never steps on values from the abandoned run.
   From setup on these values replace the object card's (and any per-world randomization of those fields).
   On SPH (WP13 review of step 4): SPH has no yield stress, so an SPH "solid" is a liquid of viscosity μ_s and
   creeps; the weiler2018 pair viscosity is the arithmetic mean (μ_i + μ_j)/2, so a solid neighbour dominates a melt
   pair; and large μ_s / μ_l ratios slow its CG solve (100 iterations, tolerance 1e-3): watch for UNCONVERGED.

Heater modes: ``"temperature"`` (a thermostat at ``heater_values`` K), ``"power"`` (``heater_values`` W shared by
contact area) and ``"body"``: the heater shape's body is a WP13 ``Thermal`` body. Each particle exchanges
h·A·(T_body − T_i) (A its footprint) in whole µJ per step, and the body's ``external_heat`` gets exactly the opposite.
Call ``link_thermal(sim.plugins)`` after building; ``Thermal`` must come before this plug-in in the list. The host
runs after-hooks in list order, so Thermal books each exchange's heat in its next update: one exchange of lag, exact
energy. ``external_heat`` is an input, not state, so a rollback can lose or repeat one exchange of body-heater heat.

A particle within its radius plus ``heater_gap`` of a heater face touches it with the share of its footprint over the
face that no lower particle covers (``particle_kernels.touch_share``). The default gap, a quarter spacing, suits SPH
(its bottom layer rests ~0.1 mm beyond a radius: at 0, contact fell from 28 % to 1 % in 1.5 s); MPM needs a voxel.

Thermal particles are chosen per group by a template particle range (the same in every world) with a material from
``phase_materials.MATERIALS`` and a start temperature (at most the boiling point). State, float64:
- [W, P]: ``enthalpy``, ``temperature``, ``liquid`` (φ), ``boiled`` (boiled energy [J]), ``done`` (doneness), and the
  ledger ``heated`` / ``cooled`` (each particle's heater gain and ambient loss [J]);
- [W, K]: ``heater_idle`` [J], power a power heater put nowhere because no particle taking heat touched it;
- [W] int32: ``neighbor_overflow_max``, the most particles that overflowed their rows in any one exchange.
``read()`` adds per world ``stored_J`` (Σ m H, including evaporated particles' m·H_b), ``heat_in_J``, ``heat_out_J``,
``boiled_J``, ``evaporated_kg``, ``neighbor_overflow`` and ``evaporated_in_solver``; per particle ``vapor``
(boiled / (m L_v)) and ``taking_heat`` (the kernels' predicate); per heater ``heater_idle_J``. Inputs:
``heater_values`` [W, K], each heater's temperature [K] or power [W]; ``ambient`` [W], the air temperature [K] for
convection (an oven per world).

Only particles that take heat exchange it: ACTIVE and not evaporated (one predicate in every pass). A spare (SPH parks
spares at one point with ACTIVE cleared) keeps its enthalpy, so it starts at its group's ``initial`` temperature when
it is activated, unless ``set_temperature`` changes it. Conservation needs complete neighbour rows: an overflowing row
drops pairs one way (``neighbor_overflow`` is 0 for a conservative run).

**Evaporated particles stay in the solver** until WP9's ``retire(mask)`` lands: still ACTIVE, with their mass and
place, but thermally gone. Next to a heater such a layer holds the liquid above it out of contact, so heating stops
(WP13's probe: water at 372.15 K on a 450 K plate stalls once the bottom layer, 25 % of the mass, has boiled off, the
rest staying at 372.2 K). ``evaporated_in_solver`` counts them per world; a scene that boils off must check it is 0.

Example (by object name: ``particle_scene.for_scene``; the Scene API: docs/guide/heat.md)::

    therm = ParticleThermal(materials=("water",), initial=(293.15,), heaters=("plate",), heater_values=(353.15,),
                            heater_modes=("temperature",))
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
import warp as wp

from alloy._vendor import newton

from alloy.bully.kernels import world_csr
from alloy.plugins.base import Forces, ForcePlugin
from alloy.plugins.errors import PluginError
from alloy.plugins.thermal import particle_kernels as K
from alloy.plugins.thermal import particle_area as PA
from alloy.plugins.thermal import particle_area_kernels as AK
from alloy.plugins.thermal import phase_kernels as PK
from alloy.plugins.thermal.phase_materials import PhaseMaterial, material

log = logging.getLogger("alloy.plugins")

MODES = {"temperature": 0, "power": 1, "body": 2}
FEEDBACK = ("auto", "off")
STABILITY = 0.1  # explicit conduction: dt <= STABILITY · ρ c hs² / k (Cleary and Monaghan 1999)
GAP_SPACINGS = 0.25  # default heater_gap: SPH rests its bottom layer ~0.1 mm beyond a radius off a floor (step 5)


class ParticleThermal(ForcePlugin):
    """Per-particle temperature with conduction, heaters and the enthalpy method (see the module docstring)."""

    name = "particle_thermal"
    summary = "per-particle temperature (heat v2): conduction, heaters, enthalpy phase change"
    needs_contacts: ClassVar[bool] = False
    # Harness scene (XPBD on still particles, an almost free step): 2.9-3.0x (CPU, 600 slots, 11-07); convection adds
    # the cone pass: 4.1-4.4x on that mostly-surface scene. Against MPM / SPH: the GPU smoke rows' cost_ratio.
    speed_budget: ClassVar[float] = 4.0

    @dataclass
    class Config:
        materials: tuple[str, ...] = ("water",)            # one per group (phase_materials.MATERIALS)
        ranges: tuple[tuple[int, int], ...] = ((0, -1),)   # template particle range [start, end) per group; -1: end
        initial: tuple[float, ...] = (293.15,)             # K, one per group
        support: float = 0.0                               # m, kernel support (0: 2.2 x the mean spacing (m/ρ)^⅓)
        neighbor_capacity: int = 64
        heaters: tuple[str, ...] = ()                      # box shape labels
        heater_modes: tuple[str, ...] = ()                 # "temperature" | "power" | "body", one per heater
        heater_values: tuple[float, ...] = ()              # K or W per heater (input "heater_values"; body: unused)
        heater_conductance: float = 1000.0                 # W/(m²·K), particle to a temperature heater
        heater_efficiency: float = 1.0                     # fraction of a power heater's power that goes in
        heater_gap: float | None = None                    # m beyond the radius that still touches (None: ¼ spacing)
        ambient: float = 293.15                            # K, every world's start value of the input "ambient" [W]
        convection: float = 0.0                            # W/(m²·K) on exposed surfaces (0: off)
        time_scale: float = 1.0                            # thermal time runs this many times faster (design §4)
        power_density: tuple[float, ...] = ()              # W/kg volumetric heat per group (microwave-like; () = 0)
        feedback: str = "auto"                             # "auto": write flow values into sph / mpm arrays; "off"
        buoyancy: bool = False                             # hot liquid rises (for_scene: plugins.thermal.buoyancy)

        def __post_init__(self) -> None:
            n = len(self.materials)
            if n == 0 or len(self.ranges) != n or len(self.initial) != n:
                raise ValueError("materials, ranges and initial need one entry per group")
            if self.power_density and len(self.power_density) != n:
                raise ValueError("power_density needs one entry per group (or none)")
            if not len(self.heaters) == len(self.heater_modes) == len(self.heater_values):
                raise ValueError("heaters, heater_modes and heater_values need one entry per heater")
            bad = [m for m in self.heater_modes if m not in MODES]
            if bad:
                raise ValueError(f"heater_modes {bad}: use {sorted(MODES)}")
            if self.feedback not in FEEDBACK:
                raise ValueError(f"feedback {self.feedback!r}: use {FEEDBACK}")
            if "body" in self.heater_modes and self.time_scale != 1.0:
                raise ValueError("body heaters need time_scale = 1 (the Thermal body runs in real time)")
            if min(self.heater_conductance, self.heater_efficiency, self.heater_gap or 0.0, self.convection,
                   self.support) < 0.0 or self.time_scale <= 0.0 or min(self.initial) <= 0.0:
                raise ValueError("conductance, efficiency, gap, convection, support >= 0; time_scale > 0; "
                                 "initial temperatures > 0 K")
            for name, t0 in zip(self.materials, self.initial):
                if t0 > material(name).t_boil:
                    raise ValueError(f"initial {t0} K for {name} is above its boiling point "
                                     f"{material(name).t_boil} K (it would flash to vapour on the first step)")

    # --- setup ---------------------------------------------------------------------------------------------------

    def setup(self, ctx: Any) -> None:
        c, model = self.config, ctx.model
        self._W = ctx.world_count
        gidx, group = self._slots(model)
        self.P = gidx.size // self._W
        self._group = group[: self.P]  # slot -> config group (world 0; the same in every world)
        self._mats = [material(n) for n in dict.fromkeys(c.materials)]
        mat_id = np.asarray([self._mats.index(material(c.materials[g])) for g in group], np.int32)
        self._gidx, self._mat = ctx.array(gidx, wp.int32), ctx.array(mat_id, wp.int32)
        slot_of = np.full(model.particle_count, -1, np.int32)
        slot_of[gidx] = np.arange(gidx.size, dtype=np.int32)
        self._slot_of = ctx.array(slot_of, wp.int32)
        self._phase = ctx.array(np.where(slot_of >= 0, 1, -1).astype(np.int8), wp.int8)
        self._mt = self._table(ctx)
        h0 = np.concatenate([[material(c.materials[g]).enthalpy(c.initial[g])] for g in group]).astype(np.float64)
        self.enthalpy = self.add_state("enthalpy", ctx.array(h0.reshape(self._W, self.P), wp.float64))
        self.temperature = self.add_state("temperature", ctx.zeros((self._W, self.P), wp.float64))
        self.liquid = self.add_state("liquid", ctx.zeros((self._W, self.P), wp.float64))
        self.heated = self.add_state("heated", ctx.zeros((self._W, self.P), wp.float64))
        self.cooled = self.add_state("cooled", ctx.zeros((self._W, self.P), wp.float64))
        self.boiled = self.add_state("boiled", ctx.zeros((self._W, self.P), wp.float64))
        self.done = self.add_state("done", ctx.zeros((self._W, self.P), wp.float64))
        self._flat = {k: a.reshape((-1,)) for k, a in (("h", self.enthalpy), ("t", self.temperature),
                                                       ("phi", self.liquid), ("in", self.heated),
                                                       ("out", self.cooled), ("boil", self.boiled),
                                                       ("done", self.done))}
        self._rate = ctx.zeros(gidx.size, wp.float64)
        self._moment = ctx.zeros(gidx.size, wp.mat33d)
        self._mass = model.particle_mass.numpy()[gidx].astype(np.float64)
        self._slot_mass = ctx.array(self._mass, wp.float64)
        qv = np.asarray(c.power_density or [0.0] * len(c.materials), np.float64)
        self._power_density = ctx.array(qv[group], wp.float64)
        rho = np.asarray([material(c.materials[g]).density for g in group], np.float64)
        self._face = (self._mass / rho) ** (2.0 / 3.0)  # footprint [m²] (particle_kernels doc: Areas)
        self._exposure_arr = ctx.zeros(gidx.size, wp.float64)
        self._setup_neighbors(ctx, gidx)
        self._setup_heaters(ctx)
        self._setup_structs(ctx)
        self._setup_flow(ctx)
        wp.launch(K.derive, dim=gidx.size, device=ctx.device,
                  inputs=[self._flat["h"], self._mat, self._mt], outputs=[self._flat["t"], self._flat["phi"]])
        self._write_flow()  # before the solvers are built: MPM fixes has_viscosity and its stiffness path at build
        self._report_flow(model)
        self._check_density(model, gidx, rho)

    def _slots(self, model: newton.Model) -> tuple[np.ndarray, np.ndarray]:
        """(global particle id of every slot, world-major; group of every slot) from the template ranges."""
        start, index = world_csr(model.particle_world.numpy(), model.world_count)
        per_world, groups = [], None
        for w in range(model.world_count):
            items = index[start[w]: start[w + 1]]
            ids, grp = [], []
            for g, (a, b) in enumerate(self.config.ranges):
                b = items.size if b < 0 else b
                if not 0 <= a < b <= items.size:
                    raise PluginError(f"{self.label}: range {(a, b)} outside world {w}'s {items.size} particles")
                ids.append(items[a:b])
                grp.append(np.full(b - a, g, np.int32))
            per_world.append(np.concatenate(ids))
            groups = np.concatenate(grp) if groups is None else groups
        if len({p.size for p in per_world}) != 1 or per_world[0].size == 0:
            raise PluginError(f"{self.label}: every world needs the same thermal particle count (> 0)")
        if len(np.unique(np.concatenate(per_world))) != sum(p.size for p in per_world):
            raise PluginError(f"{self.label}: particle ranges overlap")
        return np.concatenate(per_world).astype(np.int64), np.tile(groups, model.world_count)

    def _table(self, ctx: Any) -> K.MatTable:
        mt = K.MatTable()
        cols = {"density": "density", "c_s": "c_solid", "c_l": "c_liquid", "k_s": "k_solid", "k_l": "k_liquid",
                "t_s": "t_solidus", "t_l": "t_liquidus"}
        for field, attr in cols.items():
            setattr(mt, field, ctx.array([getattr(m, attr) for m in self._mats], wp.float64))
        mt.h_l = ctx.array([m.h_liquidus for m in self._mats], wp.float64)
        mt.h_b = ctx.array([min(m.h_boil, K.NO_BOIL) for m in self._mats], wp.float64)
        mt.l_v = ctx.array([m.latent_vapor for m in self._mats], wp.float64)
        mt.kin_a = ctx.array([m.doneness_rate for m in self._mats], wp.float64)
        mt.kin_ea = ctx.array([m.doneness_energy for m in self._mats], wp.float64)
        return mt

    def _setup_neighbors(self, ctx: Any, gidx: np.ndarray) -> None:
        from alloy.solvers.sph.neighbors import NeighborLists  # noqa: PLC0415 - the SPH package's sorted rows

        model, c = ctx.model, self.config
        spacing = float(np.sqrt(self._face).mean())  # (m/ρ)^(1/3): the lattice spacing, whatever the collision radius
        self.support = float(c.support) if c.support > 0.0 else 2.2 * spacing
        self.heater_gap = float(c.heater_gap) if c.heater_gap is not None else GAP_SPACINGS * spacing
        self._nbr = NeighborLists(model.particle_count, c.neighbor_capacity, self.support, ctx.device)
        self._overflow = ctx.zeros(self._W, wp.int32)  # this exchange
        self._overflow_max = self.add_state("neighbor_overflow_max", ctx.zeros(self._W, wp.int32))
        self._nbr.build(model.particle_q, model.particle_flags, self._phase, model.particle_world, self._overflow)
        # worst footprint per mass [m²/kg] for check_dt; convection exposure is at most CAP footprints (a lone particle)
        self._exposure = float((self._face / self._mass).max())
        ratio = self.support / spacing
        self._area = wp.vec3d(PA.FLOOR, PA.fill_calibration(ratio), PA.face_calibration(ratio))  # (δ, κ_F, κ_C)
        self._dirs = ctx.array(PA.directions().astype(np.float32), wp.vec3)
        self._overflow.zero_()

    def _setup_heaters(self, ctx: Any) -> None:
        c, model = self.config, ctx.model
        n = len(c.heaters)
        shapes = np.full(max(self._W * n, 1), -1, np.int32)
        kinds = model.shape_type.numpy()
        for k, label in enumerate(c.heaters):
            ids = ctx.shapes(label).numpy()
            if (kinds[ids] != int(newton.GeoType.BOX)).any():
                raise PluginError(f"{self.label}: heater {label!r} must be a box shape (v2 heaters are boxes)")
            shapes[np.arange(self._W) * n + k] = ids
        ht = K.Heaters()
        ht.shape = ctx.array(shapes, wp.int32)
        ht.mode = ctx.array([MODES[m] for m in c.heater_modes] or [0], wp.int32)
        values = np.tile(np.asarray(c.heater_values or [0.0], np.float64), (self._W, 1))
        ht.value = self.add_input("heater_values", ctx.array(values, wp.float64))
        ht.area = ctx.zeros(max(self._W * n, 1), wp.int64)
        ht.conductance, ht.efficiency = wp.float64(c.heater_conductance), wp.float64(c.heater_efficiency)
        ht.gap = self.heater_gap
        ht.count = n
        ht.body_slot = ctx.full(max(self._W * n, 1), -1, wp.int32)
        ht.body_temp, ht.body_heat = ctx.zeros(1, wp.float64), ctx.zeros(1, wp.int64)
        self.body_residual = self.add_state("body_residual", ctx.zeros((self._W, self.P * max(n, 1)), wp.float64))
        ht.residual = self.body_residual.reshape((-1,))
        self._ht, self.n_heaters, self._heater_shapes = ht, n, shapes[: self._W * n].reshape(self._W, n)
        self._body_heaters = [k for k, m in enumerate(c.heater_modes) if m == "body"]
        self.heater_idle = self.add_state("heater_idle", ctx.zeros((self._W, max(n, 1)), wp.float64))
        self._idle_flat = self.heater_idle.reshape((-1,))
        self._linked = False

    def _setup_flow(self, ctx: Any) -> None:
        """Flow tables and the solver arrays to write (module doc, item 4)."""
        model, mats = ctx.model, self._mats
        ft = PK.FlowTable()
        cols = {"mu_l": "mu_liquid", "mu_e": "mu_energy", "mu_d": "mu_done", "mu_s": "mu_solid", "n": "yield_exponent",
                "tau_s": "yield_solid", "fric_s": "friction_solid", "e_s": "young_solid", "e_l": "young_liquid",
                "t_s": "t_solidus", "t_l": "t_liquidus"}
        for field, attr in cols.items():
            setattr(ft, field, ctx.array([getattr(m, attr) for m in mats], wp.float64))
        out, dummy = PK.FlowTargets(), ctx.zeros(1, wp.float32)
        out.sph_viscosity = out.mpm_viscosity = out.mpm_yield = out.mpm_friction = out.mpm_young = dummy
        write = 0
        if self.config.feedback == "auto" and any(m.flows for m in mats):
            sph, mpm = getattr(model, "sph", None), getattr(model, "mpm", None)
            if getattr(sph, "viscosity", None) is not None:
                out.sph_viscosity, write = sph.viscosity, write | int(PK.WRITE_SPH)
            if getattr(mpm, "yield_stress", None) is not None:
                out.mpm_viscosity, out.mpm_yield, out.mpm_young = mpm.viscosity, mpm.yield_stress, mpm.young_modulus
                out.mpm_friction = mpm.friction
                write |= int(PK.WRITE_MPM)
        out.write = write
        self._ft, self._targets = ft, out
        log.info("%s: flow feedback writes %s", self.label,
                 [k for k, bit in (("sph", PK.WRITE_SPH), ("mpm", PK.WRITE_MPM)) if write & int(bit)] or "nothing")

    def _check_density(self, model: newton.Model, gidx: np.ndarray, rho: np.ndarray) -> None:
        """Areas use the table's ρ with the model's mass: warn when the solver's density says otherwise (a mismatch
        scales every footprint by (ρ_model/ρ_table)^(2/3), and F reads density directly: 9 % denser read 0.91 of the
        area), and when group spacings differ (κ is calibrated once, at the mean: 1.22x coarser read 0.91)."""
        rest = getattr(getattr(model, "sph", None), "rest_density", None)
        checks = [("SPH rest density", rest.numpy()[gidx])] if rest is not None else []
        if getattr(model, "mpm", None) is not None:  # Newton's MPM: ρ = m / (2r)³ (a cube of side 2r per particle)
            checks.append(("MPM density m/(2r)³", self._mass / (2.0 * model.particle_radius.numpy()[gidx]) ** 3))
        for what, got in checks:
            off = float(np.abs(got / rho - 1.0).max())
            if off > 0.01:
                log.warning("%s: %s differs from the heat table's by up to %.1f %%: footprints and exposure use the "
                            "table's density", self.label, what, 100.0 * off)
        group, spacing = self._group, np.sqrt(self._face[: self.P])
        means = [float(spacing[group == g].mean()) for g in np.unique(group)]
        if max(means) > 1.1 * min(means):
            log.warning("%s: group spacings differ by %.0f %%: exposure is calibrated at the mean spacing and reads a "
                        "coarser group low (1.22x coarser: 0.91 of its area)", self.label, 100.0 * (max(means) /
                                                                                             min(means) - 1.0))

    def _report_flow(self, model: newton.Model) -> None:
        """Log whose card values the flow feedback replaces, and warn when SPH would ignore it: SPH picks its viscosity
        mode once at build from the largest start μ, and at <= 1e-2 Pa·s ("auto" -> xsph) per-particle μ is unused."""
        if not self._targets.write:
            return
        flows = np.asarray([m.flows for m in self._mats])[self._mat.numpy()]
        log.info("%s: %d particles' card viscosity / yield / friction / E are replaced by the flow law",
                 self.label, int(flows.sum()))
        if self._targets.write & int(PK.WRITE_SPH):
            from alloy.solvers.sph.config import VISCOUS_ABOVE  # noqa: PLC0415 - the SPH package's router rule

            mu = model.sph.viscosity.numpy()
            if float(mu.max()) <= VISCOUS_ABOVE:
                log.warning("%s: the largest start viscosity is %.3g Pa·s <= %.3g: if the SPH entry's viscosity is "
                            "'auto' (its default) it resolves to xsph, which ignores per-particle μ; set "
                            "viscosity='weiler2018' on the entry", self.label, float(mu.max()), VISCOUS_ABOVE)

    def _write_flow(self) -> None:
        if self._targets.write:
            f = self._flat
            wp.launch(PK.write_flow, dim=self._gidx.shape[0], device=self.ctx.device,
                      inputs=[self._sl, f["t"], f["phi"], f["done"], self._ft, self._targets])

    def link_thermal(self, host: Any, label: str = "thermal") -> None:
        """Connect the ``"body"`` heaters to WP13's ``Thermal`` plug-in ``label`` in ``host`` (``sim.plugins``),
        which must come before this plug-in. Each body heater's body must be one of Thermal's ``bodies``."""
        order = list(host.plugins)
        if label not in order:
            raise PluginError(f"{self.label}: no plug-in {label!r} to link; plug-ins: {order}")
        if order.index(label) > order.index(self.label):
            raise PluginError(f"{self.label}: put {label!r} before {self.label!r} in the plug-in list (its update "
                              f"must consume the body heat of the previous exchange)")
        thermal = host.plugins[label]
        names = list(thermal.config.bodies)
        ids = np.stack([self.ctx.bodies(b).numpy() for b in names], axis=1)  # [W, B] global body ids
        shape_body = self.ctx.model.shape_body.numpy()
        slots = np.full(max(self._W * self.n_heaters, 1), -1, np.int32)
        for k in self._body_heaters:
            for w in range(self._W):
                hit = np.flatnonzero(ids[w] == shape_body[self._heater_shapes[w, k]])
                if hit.size == 0:
                    raise PluginError(f"{self.label}: body heater {self.config.heaters[k]!r} is not on one of "
                                      f"{label!r}'s bodies {names}")
                slots[w * self.n_heaters + k] = w * len(names) + int(hit[0])
        self._ht.body_slot.assign(slots)
        self._ht.body_temp = thermal.temperature.reshape((-1,))
        self._ht.body_heat = thermal.external_heat.reshape((-1,))
        self._linked = True

    def _setup_structs(self, ctx: Any) -> None:
        """The kernels' grouped arguments. Fields that change per exchange (positions, body poses, dt) are set in
        ``update``; every other array is fixed for the plug-in's life (state restores write in place)."""
        c, model, f = self.config, ctx.model, self._flat
        sl = K.Slots()
        sl.gidx, sl.slot_of, sl.mat = self._gidx, self._slot_of, self._mat
        sl.mass, sl.power_density, sl.per_world = self._slot_mass, self._power_density, self.P
        sl.face = ctx.array(self._face, wp.float64)
        cl = K.Cloud()
        cl.q, cl.mass, cl.radius, cl.flags = model.particle_q, model.particle_mass, model.particle_radius, \
            model.particle_flags
        cl.nbr, cl.count, cl.hs = self._nbr.nbr, self._nbr.count, 0.5 * self.support
        sh = K.Shapes()
        sh.body, sh.xf, sh.scale = model.shape_body, model.shape_transform, model.shape_scale
        sh.body_q = model.shape_transform  # replaced by the state's body_q each exchange when there are bodies
        fd = K.Field()
        fd.temp, fd.liquid, fd.moment, fd.boiled = f["t"], f["phi"], self._moment, f["boil"]
        fd.exposure = self._exposure_arr
        env = K.Env()
        env.ambient = self.add_input("ambient", ctx.array(np.full(self._W, c.ambient), wp.float64))
        env.convection = wp.float64(c.convection)
        env.dt = wp.float64(0.0)
        self._sl, self._cl, self._sh, self._fd, self._env = sl, cl, sh, fd, env

    # --- step ----------------------------------------------------------------------------------------------------

    def check_dt(self, dt: float) -> None:
        c, hs = self.config, 0.5 * self.support
        step = dt * c.time_scale
        for m in self._mats:
            limit = STABILITY * m.density * min(m.c_solid, m.c_liquid) * hs * hs / max(m.k_solid, m.k_liquid)
            if step > limit:
                raise PluginError(f"{self.label}: thermal step {step:.3g} s exceeds the conduction limit {limit:.3g} s "
                                  f"for {m.name} (support {self.support:.3g} m); lower time_scale or dt, or raise "
                                  f"support")
        h = max(c.heater_conductance if self.n_heaters else 0.0, PA.CAP * c.convection)
        c_min = min(min(m.c_solid, m.c_liquid) for m in self._mats)
        robin = step * h * self._exposure / c_min
        if robin >= 1.0:
            raise PluginError(f"{self.label}: heater / convection coupling dt·h·A/(m·c) = {robin:.3g} >= 1 is "
                              f"unstable; lower heater_conductance, convection, time_scale or dt")

    @property
    def applies_forces(self) -> bool:
        """The host calls ``apply`` before each solver step only when the flow arrays need writing (known before
        setup, when the host asks)."""
        return self.config.feedback == "auto" and any(material(n).flows for n in self.config.materials)

    def apply(self, state: newton.State, forces: Forces, dt: float) -> None:
        """Before the solver step: the flow arrays from the current (possibly just restored) T and φ. No forces."""
        self._write_flow()

    def update(self, state_in: newton.State, state_out: newton.State, dt: float) -> None:
        ctx, m, f = self.ctx, self.ctx.model, self._flat
        dev, n = ctx.device, self._gidx.shape[0]
        q = state_out.particle_q
        self._cl.q = q
        if m.body_count:
            self._sh.body_q = state_out.body_q
        if self._body_heaters and not self._linked:
            raise PluginError(f"{self.label}: body heaters need link_thermal(sim.plugins) before the first step")
        self._env.dt = wp.float64(float(dt) * self.config.time_scale)
        wp.launch(K.set_phase, dim=n, device=dev, inputs=[self._sl, self._mt, f["boil"]], outputs=[self._phase])
        self._overflow.zero_()
        self._nbr.build(q, m.particle_flags, self._phase, m.particle_world, self._overflow)
        wp.launch(K.keep_max, dim=self._W, device=dev, inputs=[self._overflow], outputs=[self._overflow_max])
        if self.n_heaters:
            self._ht.area.zero_()
            wp.launch(K.heater_area, dim=n, device=dev, inputs=[self._sl, self._cl, self._sh, self._ht, self._mt,
                                                                f["boil"]])
            wp.launch(K.heater_idle, dim=self._W * self.n_heaters, device=dev, inputs=[self._ht, self._env.dt],
                      outputs=[self._idle_flat])
        area = [self._dirs, float(np.cos(np.radians(PA.CONE_DEG))), self._area, int(self.config.convection > 0.0)]
        wp.launch(AK.surface_moment, dim=n, device=dev, inputs=[self._sl, self._cl, self._mt, *area],
                  outputs=[self._moment, self._exposure_arr])
        wp.launch(K.rate, dim=n, device=dev, inputs=[self._sl, self._fd, self._cl, self._mt, self._sh, self._ht,
                                                      self._env], outputs=[self._rate, f["in"], f["out"]])
        wp.launch(K.update, dim=n, device=dev, inputs=[self._rate, self._env.dt, self._sl, self._mt, self._cl],
                  outputs=[f["h"], f["t"], f["phi"], f["boil"], f["done"]])
        self._write_flow()

    def set_temperature(self, values: Any) -> None:
        """Set every slot's temperature [K] outside the step: ``[W, P]``, ``[P]`` (every world) or a scalar."""
        t = np.broadcast_to(np.asarray(values, np.float64), (self._W, self.P))
        mat = self._mat.numpy().reshape(self._W, self.P)
        t_boil = np.asarray([m.t_boil for m in self._mats])[mat]
        if (t > t_boil).any():
            raise PluginError(f"{self.label}: set_temperature above the boiling point (it would flash to vapour)")
        h = np.empty((self._W, self.P), np.float64)
        for i, m in enumerate(self._mats):
            sel = mat == i
            h[sel] = m.enthalpy(t[sel])
        self.enthalpy.assign(h)
        wp.launch(K.derive, dim=self._W * self.P, device=self.ctx.device,
                  inputs=[self._flat["h"], self._mat, self._mt], outputs=[self._flat["t"], self._flat["phi"]])
        self._write_flow()

    # --- readout -------------------------------------------------------------------------------------------------

    def read(self) -> dict[str, np.ndarray]:
        out = super().read()
        out["stored_J"] = (out["enthalpy"] * self._mass.reshape(self._W, self.P)).sum(axis=1)
        out["heat_in_J"] = out["heated"].sum(axis=1)
        out["heat_out_J"] = out["cooled"].sum(axis=1)
        out["exposure"] = self._exposure_arr.numpy().reshape(self._W, self.P)  # m², last exchange (0: no convection)
        out["heater_area_m2"] = (self._ht.area.numpy()[: self._W * self.n_heaters].reshape(self._W, self.n_heaters)
                                 / float(K.AREA_FP))  # Σ footprints touching each heater, the last exchange
        out["boiled_J"] = out["boiled"].sum(axis=1)
        lv = np.asarray([m.latent_vapor for m in self._mats])[self._mat.numpy()].reshape(self._W, self.P)
        mass = self._mass.reshape(self._W, self.P)
        with np.errstate(divide="ignore", invalid="ignore"):
            vapor = np.where(lv > 0.0, out["boiled"] / (mass * lv), 0.0)
        out["vapor"] = vapor
        out["evaporated_kg"] = (np.minimum(vapor, 1.0) * mass).sum(axis=1)
        out["neighbor_overflow"] = out["neighbor_overflow_max"].copy()
        live = (self.ctx.model.particle_flags.numpy()[self._gidx.numpy()] & int(newton.ParticleFlags.ACTIVE)) != 0
        out["evaporated_in_solver"] = ((vapor >= 1.0) & live.reshape(self._W, self.P)).sum(axis=1)
        gone = (lv > 0.0) & (out["boiled"] >= mass * lv)  # particle_kernels.takes_heat, per particle
        out["taking_heat"] = live.reshape(self._W, self.P) & ~gone
        out["heater_idle_J"] = out["heater_idle"][:, : self.n_heaters].copy()
        return out

    def materials(self) -> list[PhaseMaterial]:
        """The distinct materials, in table order (``MatTable`` index)."""
        return list(self._mats)

    def laws(self):
        from alloy.plugins.thermal.particle_area_laws import area_laws  # noqa: PLC0415 - laws import this module
        from alloy.plugins.thermal.particle_laws import particle_laws  # noqa: PLC0415
        from alloy.plugins.thermal.phase_laws import phase_laws  # noqa: PLC0415

        return particle_laws() + area_laws() + phase_laws()

    def test_scene(self):
        from alloy.plugins.thermal.particle_laws import busy_scene  # noqa: PLC0415

        return busy_scene()
