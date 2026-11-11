# SPDX-License-Identifier: Apache-2.0
"""SolverVBDAlloy: Newton's SolverVBD with static-shape Divide-and-Truncate under self-contact (SCR-SC8-3) and a public
per-colour callback in two slots (SCR-SC11-1, SCR-SC11-2).

Static-shape DAT. With `particle_enable_self_contact` on (and Newton's own rigid-soft DAT off), every
penetration-free truncation first truncates the particles of each static-shape contact record against that shape's
division plane (`static_dat.truncate_against_static_shapes`); Newton's truncation then applies the self-contact planes
and the isotropic bound. Without self-contact Newton truncates nothing, there is no asymmetry, and the class is bitwise
the stock solver.

Per-colour callback (`add_colour_callback`). Newton's particle iteration zeroes `particle_forces` and
`particle_hessians` once, then for each colour accumulates attachment, contact, spring and self-contact terms into them
and solves that colour's vertices, then truncates. A registered callback runs once per colour in every particle
iteration, in one of two slots:
- ``when="before_accumulate"`` (default): before that colour's accumulation. Colour 0 runs right after the
  iteration's zeroing of `particle_hessians` (Newton zeroes `particle_forces` just before); colour c >= 1 right after
  colour c-1's penetration-free truncation, so it sees the updated iterate. For additive terms.
- ``when="after_accumulate"``: after Newton has accumulated that colour's terms (contacts and self-contact included)
  and right before its solve (`solve_elasticity` or the tiled kernel). Newton reads `use_particle_tile_solve` exactly
  once per colour at that point, and the slot fires on that read; `after_accumulate_hook_ok()` checks the order in
  Newton's source (the tripwire) and the slot refuses to register when it fails. For callbacks that need Newton's terms,
  e.g. one that sets a colour's force and Hessian so the solve applies its own step.
A callback may write force and Hessian contributions for the particles of that colour into `particle_forces` /
`particle_hessians`, using launches of fixed size: graph capture records them in order, so a captured step replays
them. With no callback registered nothing is installed and nothing extra runs (bitwise the stock loop).
`tests/solvers/vbd/test_colour_callback.py` pins both slots against Newton's loop.
Known limit (Newton ac7f7830 and main): in a deterministic Warp mode with contacts passed to `step`, Newton's
per-particle contact gather writes `particle_forces[i] = force` for every particle of the colour, discarding what was
accumulated before it (callback contributions, and Newton's own particle-body attachments). A step in that
configuration raises instead of silently dropping the callback; the guard lifts when the gather accumulates.

Opt-in accelerators (`accel.py`), off by default (off, the class is the stock loop):
- ``alloy_initial_guess="adaptive"``: VBD's adaptive initial guess (a resting sheet is a fixed point at any iteration
  count; free fall and pinned pieces keep Newton's start bit for bit).
- ``alloy_chebyshev_rho``: Chebyshev-accelerated particle iterations with that spectral radius estimate. Use
  rho <= 0.95: at 0.97 the A17 film (15 um, the stiffest membrane in tests/solvers/vbd) is over-relaxed and does not
  converge (2.5e-3 m off a 4,000-iteration solve at 30 iterations, 3.4e-4 at 2,000), while 0.9-0.95 converge there
  and faster than the stock loop (0.9 is the tested value).
- ``alloy_chebyshev_hold_self_contact=True``: particles of a self-contact pair (fold plies) are not extrapolated
  either (known gap cloth-fold-unsettled-default).

Takes SolverVBD's arguments unchanged, plus those three; adapters and estimators registered for SolverVBD resolve by MRO.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

import warp as wp

from alloy._vendor import newton
from alloy._vendor.newton._src.solvers.vbd.particle_vbd_kernels import gather_particle_body_contact_force_and_hessian as _gather

from alloy.solvers.plastic.tets import TetPlastic

from .hysteresis import BendHysteresis
from .accel import AccelConfig, AccelState
from .static_dat import truncate_against_static_shapes

ColourCallback = Callable[[Any, int, newton.State, Any], None]
"""fn(solver, colour, state, contacts): `state.particle_q` holds the current iterate; contacts may be None."""
GATHER_OVERWRITES = "particle_forces[particle_index] = force" in inspect.getsource(_gather.func)
COLOUR_SLOTS = ("before_accumulate", "after_accumulate")
_TILE_FLAG = "self.use_particle_tile_solve"


def after_accumulate_hook_ok(src: str | None = None) -> bool:
    """True when Newton's particle iteration reads `use_particle_tile_solve` once per colour, after its last
    accumulation and before the colour's solve (the after_accumulate slot's anchor)."""
    src = inspect.getsource(newton.solvers.SolverVBD._solve_particle_iteration) if src is None else src
    loop = src.find("for color in range(")
    flag = src.find(_TILE_FLAG)
    last_acc = src.rfind("kernel=accumulate_")
    solve = src.find("kernel=solve_elasticity")
    return src.count(_TILE_FLAG) == 1 and 0 <= loop < last_acc < flag < solve


class SolverVBDAlloy(newton.solvers.SolverVBD):
    """SolverVBD with static-shape DAT under self-contact and a per-colour callback (module docstring)."""

    def __init__(self, model: newton.Model, *args, alloy_initial_guess: str = "inertial",
                 alloy_chebyshev_rho: float | None = None, alloy_chebyshev_hold_self_contact: bool = False,
                 **kwargs) -> None:
        cfg = AccelConfig(alloy_initial_guess, alloy_chebyshev_rho, bool(alloy_chebyshev_hold_self_contact))
        super().__init__(model, *args, **kwargs)
        cpu_owned = (self.particle_enable_self_contact and getattr(self, "collision_pipeline", None) is None
                     and not wp.get_device(self.device).is_cuda)
        self._sc_launch_full = self.particle_self_contact_evaluation_kernel_launch_size if cpu_owned else None
        n = model.particle_count
        self._accel = AccelState(cfg, n, self.device)
        for key, arr in self._accel.arrays().items():                    # snapshot names (accel module docstring)
            setattr(self, key, arr)
        self._static_dat_t = wp.ones(n, dtype=float, device=self.device) if n else None
        self._colour_callbacks: list[ColourCallback] = []
        self._colour_callbacks_after: list[ColourCallback] = []
        self._colour_external = False
        self._colour_next = -1                 # next colour to announce inside a particle iteration, -1 outside
        self._after_next = -1                  # the same for the after_accumulate slot
        self._colour_ctx: tuple[Any, Any] | None = None
        self.plastic = TetPlastic.for_model(model)   # J2 tets (SCR-WP7-15); None: none yields, the stock step
        if self.plastic is not None:                 # solver state for snapshots (tets module doc)
            self._plastic_tet_fp_inv, self._plastic_tet_ebar = self.plastic.fp_inv, self.plastic.ebar
            self._plastic_tet_poses = model.tet_poses
        self.hysteresis = BendHysteresis.for_model(model)   # KES bending hysteresis (hysteresis.py); None: none
        if self.hysteresis is not None:
            for key, arr in self.hysteresis.arrays(model).items():         # snapshot names (hysteresis module doc)
                setattr(self, key, arr)

    def step(self, state_in, state_out, control, contacts, dt: float, *, observables=None) -> None:
        """SolverVBD.step, then the J2 return of the yielding tets and the hysteretic hinges' slide on the converged
        positions."""
        super().step(state_in, state_out, control, contacts, dt, observables=observables)
        if self.plastic is not None:
            self.plastic.after_step(state_out.particle_q)
        if self.hysteresis is not None:
            self.hysteresis.after_step(self.model, state_out.particle_q)

    # -- per-colour callback (SCR-SC11-1, SCR-SC11-2) ------------------------------------------------------------
    def add_colour_callback(self, fn: ColourCallback, *, external: bool = True,
                            when: str = "before_accumulate") -> None:
        """Register ``fn(solver, colour, state, contacts)`` in slot ``when`` (module docstring). ``external``: its
        forces may have a nonzero sum over a connected component (e.g. a tool pushing a sheet); SolverVBDMomentum then
        skips its coarse translation step, which assumes every external force is in the inertia target. Pass
        ``external=False`` only for forces that sum to zero within every component."""
        if when not in COLOUR_SLOTS:
            raise ValueError(f"add_colour_callback: when={when!r}, expected one of {COLOUR_SLOTS}")
        if self.model.particle_count == 0:
            raise ValueError("add_colour_callback: the model has no particles")
        if when == "after_accumulate":
            if not after_accumulate_hook_ok():
                raise RuntimeError("add_colour_callback(when='after_accumulate'): Newton's particle iteration no "
                                   "longer reads use_particle_tile_solve once between accumulation and solve "
                                   "(vbd/base.py after_accumulate_hook_ok)")
            self._colour_callbacks_after.append(fn)
        else:
            if not self._colour_callbacks:
                self._install_colour_zero_hook()
            self._colour_callbacks.append(fn)
        self._colour_external = self._colour_external or bool(external)

    def _install_colour_zero_hook(self) -> None:
        """Announce colour 0 when Newton zeroes `particle_hessians` at the start of a particle iteration."""
        arr = self.particle_hessians
        zero = type(arr).zero_.__get__(arr)

        def zero_then_colour_0() -> None:
            zero()
            if self._colour_next == 0:
                self._announce_colour()

        arr.zero_ = zero_then_colour_0
        self._colour_hooked = arr

    def _announce_colour(self) -> None:
        c = self._colour_next
        state, contacts = self._colour_ctx
        for fn in self._colour_callbacks:
            fn(self, c, state, contacts)
        self._colour_next = c + 1

    def _announce_after(self) -> None:
        c = self._after_next
        if c >= len(self.model.particle_color_groups):
            return
        state, contacts = self._colour_ctx
        for fn in self._colour_callbacks_after:
            fn(self, c, state, contacts)
        self._after_next = c + 1

    @property
    def use_particle_tile_solve(self) -> bool:
        """Newton's flag; its once-per-colour read in the particle iteration is the after_accumulate slot."""
        if getattr(self, "_after_next", -1) >= 0:
            self._announce_after()
        return self.__dict__.get("_tile_solve", False)

    @use_particle_tile_solve.setter
    def use_particle_tile_solve(self, value: bool) -> None:
        self.__dict__["_tile_solve"] = bool(value)

    # -- opt-in accelerators (accel.py) ---------------------------------------------------------------------------
    def _initialize_particles(self, state_in, state_out, contacts, dt):
        if self._sc_launch_full is not None:           # full size until this step's detection says otherwise (S6)
            self.particle_self_contact_evaluation_kernel_launch_size = self._sc_launch_full
        acc = self._accel
        acc.begin_step(state_in, dt)
        try:
            super()._initialize_particles(state_in, state_out, contacts, dt)
        finally:
            acc.pending_dt = None
        acc.start_iterations(self, contacts)

    def _solve_particle_iteration(self, state_in, state_out, contacts, dt):
        self._iteration_with_callbacks(state_in, state_out, contacts, dt)
        acc = self._accel
        if acc.x1 is None or self.model.particle_count == 0:
            return
        if acc.mix(self, state_in):
            self._truncate_plain(state_in, contacts)
            wp.copy(state_out.particle_q, state_in.particle_q)
        acc.roll(self)

    def _collision_detection_penetration_free(self, current_state, *, reset_reference: bool = True):
        super()._collision_detection_penetration_free(current_state, reset_reference=reset_reference)
        self._size_self_contact_launches()
        acc = getattr(self, "_accel", None)                # Newton's __init__ detects before it is set
        if acc is not None:
            acc.mark_self_contact(self)                    # a mid-step detection's new pairs are held from here on

    def _size_self_contact_launches(self) -> None:
        """S6: Newton launches its per-colour self-contact accumulation and planar truncation over the whole pair buffer
        every iteration, 14 % and 9 % of a CPU cloth step (L5 drape, default), even when detection found no pair. Then
        they add nothing, so until the next detection they launch at size 0: bitwise the same. Each step starts at the
        full size (``_initialize_particles``), so a reset or snapshot restore never runs on a stale 0. CPU only (the
        counts are read on the host; a CUDA graph keeps its sizes). The isotropic per-iteration bound reads the
        self-contact flag, not this size, and stays. L5 drape on the CPU: 1.29x, DC and strain unchanged."""
        full = getattr(self, "_sc_launch_full", None)      # Newton's __init__ may detect before it is set
        if full is None:
            return
        info = self.trimesh_collision_info
        counts = (info.vertex_colliding_triangles_count, info.edge_colliding_edges_count,
                  info.triangle_colliding_vertices_count)          # the last one is not always allocated
        pairs = any(c is not None and c.numpy().any() for c in counts)
        self.particle_self_contact_evaluation_kernel_launch_size = full if pairs else 0

    @property
    def chebyshev_rho(self) -> float | None:
        """The Chebyshev rho the next step runs (None: off, as built)."""
        return self._accel.cfg.chebyshev_rho

    def set_chebyshev_rho(self, rho: float) -> None:
        """Set Chebyshev's rho for the following steps (``AccelState.set_rho``: on at build only, a pure function of the
        graph key when steps are captured)."""
        self._accel.set_rho(rho)

    def reset(self, state, world_mask=None, flags=None):
        super().reset(state, world_mask, flags)
        self._accel.clear(self.model, world_mask)
        if self.plastic is not None:
            self.plastic.reset(world_mask)
        if self.hysteresis is not None:
            self.hysteresis.reset(self.model, world_mask)

    def _truncate_plain(self, state, contacts) -> None:
        """The penetration-free truncation without the per-colour hooks (colour announce, momentum step)."""
        if self._static_dat_active(contacts):
            truncate_against_static_shapes(self, contacts, self._static_dat_t)
        newton.solvers.SolverVBD._penetration_free_truncation(self, state, contacts)

    def _iteration_with_callbacks(self, state_in, state_out, contacts, dt):
        before, after = self._colour_callbacks, self._colour_callbacks_after
        if not before and not after:
            return super()._solve_particle_iteration(state_in, state_out, contacts, dt)
        if before and GATHER_OVERWRITES and self._particle_contact_use_gather and contacts is not None:
            raise RuntimeError("per-colour callback with a deterministic Warp mode and contacts: Newton's contact "
                               "gather overwrites particle_forces, so the callback would be lost (vbd/base.py)")
        if before and self.particle_hessians is not getattr(self, "_colour_hooked", None):
            self._install_colour_zero_hook()               # Newton reallocated the array
        self._colour_next, self._after_next = (0 if before else -1), (0 if after else -1)
        self._colour_ctx = (state_in, contacts)
        try:
            super()._solve_particle_iteration(state_in, state_out, contacts, dt)
        finally:
            self._colour_next, self._after_next, self._colour_ctx = -1, -1, None

    # -- static-shape DAT (SCR-SC8-3) -----------------------------------------------------------------------------
    def _static_dat_active(self, contacts) -> bool:
        return (self._static_dat_t is not None and self.particle_enable_self_contact
                and not self.rigid_soft_enable_dat and contacts is not None and contacts.soft_contact_max > 0)

    def _penetration_free_truncation(self, state, contacts):
        if self._accel.pending_dt is not None:         # the step's first truncation: the adaptive start (accel.py)
            self._accel.apply_start(self, state)
        if self._static_dat_active(contacts):
            truncate_against_static_shapes(self, contacts, self._static_dat_t)
        super()._penetration_free_truncation(state, contacts)
        if 0 < self._colour_next < len(self.model.particle_color_groups):
            self._announce_colour()


__all__ = ["COLOUR_SLOTS", "GATHER_OVERWRITES", "ColourCallback", "SolverVBDAlloy", "after_accumulate_hook_ok"]
