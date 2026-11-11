# SPDX-License-Identifier: Apache-2.0
"""MJCF light bodies: the assembler's native contact rule on the MuJoCo entry (coordinator ruling 2026-11-05 07:19Z).

An MJCF file gives each contact a time constant (solref) whatever the mass behind it. MuJoCo scales that constraint
by the body's own mass, so a light body gets a soft contact: T12's 2.083 ms on a 12.7 g block is about 2.9e5 N/m per
row, and a fingertip strike or a pinch drives the block 6.6-10.8 mm into the plywood or a base hull (SB-7's last real
S1s; ``[internal note]``). Library objects already get `alloy.scene.native_gains` from the assembler.
This module applies the same rule to the rows the assembler does not write.

**Light-body row** (the exact test). A MuJoCo geom row, or an explicit pair row, is a light-body row when:
1. **Scope.** It is the file's own row on a loose rigid body imported from an MJCF file:
   - a geom row: the geom's body has an MJCF label (``[<file>/]worldbody/...``; Newton names the MuJoCo body with
     "/" -> "_"); its weld group's root moves on a free joint and no jointed body hangs below it (a loose object
     such as T12's blocks, not a floating-base robot, a robot link or a hinged or sliding part); the group is not
     static (see 3.); and Newton took the geom's solref from the file (``solref_mode`` RAW, or MJCF_DEFAULT where the
     file is silent), not from a material (FORCE_SPACE);
   - a pair row: the file declared it (both geoms carry the same MJCF file label: Alloy's explicit pair rows,
     `alloy.scene.mujoco_pairs`, always join two objects), and at least one of its geoms is on such a body.
   Why these bounds: the mass in 3. is the body's contact mass only for a loose rigid body (on an articulated one
   the chain moves with it), and rows Alloy writes already come from sourced materials. The library census
   (2026-11-05) found that a scope of every dynamic MJCF body and every row touching one reached 31 of 40 MuJoCo
   scenes (robots, DC mechanisms, Alloy's pair rows), against measurements made only on T12's loose blocks and the
   A8 twins. Under this scope those measurements are unchanged.
2. **Form.** Its solref is in timeconst form (tau > 0, zeta > 0). Direct-form rows (tau <= 0) are left alone.
3. **Mass and stiffness.**
   - m is the mass the contact moves: the weld group's total mass for a geom row, and for a pair row the reduced mass
     of its two weld groups. A static side (the world's weld group, a mocap group or a massless one) counts as
     infinite.
   - The native stiffness is ke = min(K (1 - dmax) / m, stable_ke(zeta, dt, dmax)), with a quarter of the cap on rk4
     entries. K = 1e6 N/m is two D22 default materials (2e6 N/m each) in series. dmax is the row's solimp[1]. dt is
     the step the entry is stepped at.
   - The row is light when ke > 1 / (zeta tau)^2: the file's time constant is softer than the material at that
     mass. Below the stability cap that is m < K (1 - dmax) (zeta tau)^2. For T12 (2.083 ms, zeta 1, dmax 0.99) that
     is m < 43 g.

A light-body row gets tau = 1 / (zeta sqrt(ke)), keeping its zeta and solimp. Heavier bodies, stiffer rows and rows
already at the cap are unchanged, and nothing gets softer.
- **Per world.** m is each world's own (WP8's finding, 2026-11-07: every world took world 0's): a world whose block
  is drawn heavier keeps the file's row. ``Rows.masses`` sums a world's ``body_mass`` over the row's weld groups.
- **MuJoCo Warp.** A kernel runs before each step, after Newton writes the step's dt: it reads the current rows and
  each world's masses (``mjw_model.body_mass``, as Newton last wrote them, a mass change included) and rewrites only
  the light ones. That makes it graph-safe and idempotent, and gains written after construction keep their zeta.
  Since it rewrites in place and only stiffens, a body made heavier after a step keeps the stiffer row.
- **MuJoCo C.** The same rule runs on the host before each `mj_step`. The CPU route
  (``alloy.route.mjc_core_models.LightRows``) writes each world model from that world's own rows and masses, and
  again from scratch when it rebuilds them after a model change.
- **Override.** The entry kwarg ``{LIGHT_KEY: False}`` turns it off for one entry (A/B runs). There is no env override.
- **Gains.** The cap fraction follows the entry's gains: rk4's quarter cap on rk4 entries and on F6 R1's implicitfast
  entries, whose gains are rk4's (``{GAINS_KEY: True}``, passed by the entry factories; WP3's review).

**Measured** (Newton 47afecd7, MEASURE candidate ``native`` with scope ``mjcf``;
``[internal note]``):
- **SB-7.** WP7's two S1s: 100007:1106 from 6.60 mm to 1.36 mm (S2) and 100052:1864 from 10.83 mm to 0.89 mm (S0).
  In 8,192 fresh T12 episodes, the episodes past 4 mm go from 23 to 4.
- **SB-12.** No verdict flips. A8.squeeze_face 0.152 -> 0.063 mm and A8.squeeze_wedge 2.153 -> 2.023 mm (the
  wedge stays a known gap).
- **Speed.** SB-6 T12 goes 1.074x -> 1.072x and SB-5 T12 164.6x -> 165.3x.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from typing import Any

import numpy as np

from alloy.scene import derive
from alloy.scene.native_gains import RK4_KE_FRACTION, STABILITY_MARGIN
from alloy.scene.pairs import DEFAULT_K

log = logging.getLogger("alloy.contact")

LIGHT_KEY = "alloy:light_body"   # MuJoCo entry kwarg: False turns the rule off for that entry
GAINS_KEY = "alloy:rk4_gains"    # MuJoCo entry kwarg: the gains were written for rk4 (F6 R1: implicitfast on them)
K_PAIR = derive.pair_stiffness(DEFAULT_K, DEFAULT_K)   # 1e6 N/m: two D22 default materials in series
MJCF_MARK = re.compile(r"(^|_)worldbody_")   # Newton's MJCF label "[<file>/]worldbody/..." as a MuJoCo body name
GEOM_FILE = re.compile(r"^(?:(.*?)/)?worldbody/")   # the same label as a MuJoCo geom name ("/" kept): its file
TABLES = ("geom_solref", "pair_solref")


@dataclass(frozen=True)
class Rows:
    """The candidate rows of one solref table: indices, the mass each moves (the template's, world 0's), its solimp
    dmax, and the weld groups behind it, so each world's own masses give its own (``masses``)."""

    table: str
    idx: np.ndarray
    mass: np.ndarray
    dmax: np.ndarray
    side1: np.ndarray     # [n] the weld root (MuJoCo body) the row moves; -1: static
    side2: np.ndarray     # [n] a pair row's other weld root; -1: static or a geom row (no second side)
    weld: np.ndarray      # [nbody] each body's weld root (body_weldid)

    def masses(self, body_mass: np.ndarray) -> np.ndarray:
        """The mass each row moves for one world's ``body_mass`` [nbody] (the rule in 3.: weld group sums, the
        reduced mass of a pair's two groups). Example: ``rows.masses(world_model.body_mass)``."""
        total = np.bincount(self.weld, weights=np.asarray(body_mass, dtype=float), minlength=self.weld.size)
        return _reduced(_group(total, self.side1), _group(total, self.side2))

    def members(self) -> tuple[np.ndarray, np.ndarray]:
        """CSR of each weld root's bodies: (start [nbody + 1], bodies) for the MuJoCo Warp kernel's group sums."""
        order = np.argsort(self.weld, kind="stable").astype(np.int32)
        start = np.searchsorted(self.weld[order], np.arange(self.weld.size + 1)).astype(np.int32)
        return start, order


def weld_masses(m: Any) -> np.ndarray:
    """Per MuJoCo body: its weld group's total mass; inf (static) for the world's group, a mocap (kinematic) group
    and a massless one (T12's robot bases: no joint, mass 0, their own weld id)."""
    weld = np.asarray(m.body_weldid)
    total = np.bincount(weld, weights=np.asarray(m.body_mass, dtype=float), minlength=m.nbody)
    out = total[weld].astype(float)
    static = (weld == 0) | (np.asarray(m.body_mocapid)[weld] >= 0) | ~(out > 0.0)
    out[static] = np.inf
    return out


def mjcf_bodies(m: Any) -> np.ndarray:
    """Per MuJoCo body: whether Newton imported it from an MJCF file (its name carries MJCF_MARK)."""
    import mujoco  # noqa: PLC0415

    return np.array([bool(MJCF_MARK.search(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or ""))
                     for b in range(m.nbody)], dtype=bool)


def loose_bodies(m: Any) -> np.ndarray:
    """Per MuJoCo body: whether its weld group's root moves on a free joint with no jointed body below it (a loose
    rigid body; not a floating-base robot, a robot link or a hinged / sliding part)."""
    import mujoco  # noqa: PLC0415

    if m.njnt == 0:                                             # nothing moves (a chart on a fixed link, occluders)
        return np.zeros(m.nbody, dtype=bool)
    root, parent = np.asarray(m.body_weldid), np.asarray(m.body_parentid)
    jnt, num = np.asarray(m.body_jntadr)[root], np.asarray(m.body_jntnum)[root]
    free = (num >= 1) & (np.asarray(m.jnt_type)[np.maximum(jnt, 0)] == int(mujoco.mjtJoint.mjJNT_FREE))
    own = np.flatnonzero(root == np.arange(m.nbody))[1:]       # weld roots below the world: each hangs on a joint
    articulated = np.zeros(m.nbody, dtype=bool)
    articulated[root[parent[own]]] = True                       # the group a jointed body hangs from
    return free & ~articulated[root]


def file_geoms(solver: Any) -> np.ndarray:
    """Per MuJoCo geom: whether Newton took its solref from the file (`solref_mode` RAW or MJCF_DEFAULT), not from
    a material (FORCE_SPACE). All False when the model carries no `solref_mode`."""
    from alloy._vendor.newton._src.solvers.mujoco.constants import SOLREF_MODE_FORCE_SPACE  # noqa: PLC0415

    m = solver.mj_model
    mode = getattr(getattr(solver.model, "mujoco", None), "solref_mode", None)
    g2s = getattr(solver, "mjc_geom_to_newton_shape", None)
    if mode is None or g2s is None:
        return np.zeros(m.ngeom, dtype=bool)
    mode, shape = mode.numpy(), g2s.numpy()[0]
    return np.array([s >= 0 and mode[s] != SOLREF_MODE_FORCE_SPACE for s in shape], dtype=bool)


def geom_files(m: Any) -> np.ndarray:
    """Per MuJoCo geom: an id of the MJCF file its label names (``[<file>/]worldbody/...``), -1 for other geoms."""
    import mujoco  # noqa: PLC0415

    ids: dict[str, int] = {}
    out = np.full(m.ngeom, -1, dtype=np.int64)
    for g in range(m.ngeom):
        hit = GEOM_FILE.match(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "")
        if hit:
            out[g] = ids.setdefault(hit.group(1) or "", len(ids))
    return out


def _group(total: np.ndarray, side: np.ndarray) -> np.ndarray:
    """Per row: its weld group's mass from ``total`` (per weld root); inf for a static side (-1) or a massless one."""
    out = np.where(side >= 0, total[np.maximum(side, 0)], np.inf)
    return np.where(out > 0.0, out, np.inf)


def _reduced(m1: np.ndarray, m2: np.ndarray) -> np.ndarray:
    inv = np.where(np.isinf(m1), 0.0, 1.0 / m1) + np.where(np.isinf(m2), 0.0, 1.0 / m2)
    with np.errstate(divide="ignore"):
        return np.where(inv > 0.0, 1.0 / np.where(inv > 0.0, inv, 1.0), np.inf)


def candidate_rows(m: Any, file_rows: np.ndarray) -> list[Rows]:
    """The geom and pair rows in scope (module docstring, 1.): the file's rows on a loose MJCF body. `file_rows` is
    `file_geoms` (per MuJoCo geom)."""
    mass, gb = weld_masses(m), np.asarray(m.geom_bodyid)
    weld = np.asarray(m.body_weldid).astype(np.int32)
    side = np.where(np.isfinite(mass), weld, -1)[gb]                 # per geom: its moving weld root, -1 static
    loose = (mjcf_bodies(m) & loose_bodies(m))[gb] & np.isfinite(mass[gb])
    geom_on = loose & file_rows
    out = [Rows("geom_solref", np.flatnonzero(geom_on).astype(np.int32), mass[gb][geom_on],
                np.asarray(m.geom_solimp, dtype=float)[geom_on, 1], side[geom_on],
                np.full(int(geom_on.sum()), -1, np.int32), weld)]
    if m.npair:
        g1, g2, f = np.asarray(m.pair_geom1), np.asarray(m.pair_geom2), geom_files(m)
        pm = _reduced(mass[gb[g1]], mass[gb[g2]])
        on = (f[g1] >= 0) & (f[g1] == f[g2]) & (loose[g1] | loose[g2]) & np.isfinite(pm)
        out.append(Rows("pair_solref", np.flatnonzero(on).astype(np.int32), pm[on],
                        np.asarray(m.pair_solimp, dtype=float)[on, 1], side[g1][on], side[g2][on], weld))
    return [r for r in out if r.idx.size]


def stable_u(zeta: np.ndarray, dmax: np.ndarray) -> np.ndarray:
    """dt * sqrt(stable_ke) (`alloy.scene.native_gains.stable_ke`, before the rk4 fraction)."""
    return 2.0 * (np.sqrt(zeta * zeta * dmax * dmax + STABILITY_MARGIN * dmax) - zeta * dmax)


def light_solref(solref: np.ndarray, mass: np.ndarray, dmax: np.ndarray, dt: float, rk4: bool,
                 k_pair: float = K_PAIR) -> np.ndarray:
    """The rule on host rows ([n, 2] timeconst solref): light rows get tau = 1 / (zeta sqrt(ke)), others are kept."""
    tau, zeta = solref[:, 0], solref[:, 1]
    ok = (tau > 0.0) & (zeta > 0.0)
    z = np.where(ok, zeta, 1.0)
    cap = stable_u(z, dmax) / dt * (math.sqrt(RK4_KE_FRACTION) if rk4 else 1.0)
    ke = np.minimum(k_pair * (1.0 - dmax) / mass, cap * cap)
    t = 1.0 / (z * np.sqrt(ke))
    out = solref.copy()
    light = ok & (t < tau)
    out[light, 0] = t[light]
    return out


def light_mass_bound(tau: float, zeta: float, dmax: float, k_pair: float = K_PAIR) -> float:
    """Below the stability cap, a row is light when its mass is under K (1 - dmax) (zeta tau)^2 [kg]."""
    return k_pair * (1.0 - dmax) * (zeta * tau) ** 2


_KERNEL: list = []


def _kernel():
    if _KERNEL:
        return _KERNEL[0]
    import warp as wp  # noqa: PLC0415

    margin = float(STABILITY_MARGIN)

    @wp.func
    def inv_group(body_mass: wp.array2d(dtype=float), w: int, root: int, start: wp.array(dtype=int),
                  bodies: wp.array(dtype=int)) -> float:
        """1 / the weld group's mass in world w; 0 for a static side (root -1) or a massless group."""
        if root < 0:
            return 0.0
        m = wp.float32(0.0)                  # uses wp, so Warp resolves this func's annotations from its closure
        for k in range(start[root], start[root + 1]):
            m += body_mass[w, bodies[k]]
        if m <= 0.0:
            return 0.0
        return 1.0 / m

    @wp.kernel
    def light_rows(solref: wp.array2d(dtype=wp.vec2), idx: wp.array(dtype=int), side1: wp.array(dtype=int),
                   side2: wp.array(dtype=int), start: wp.array(dtype=int), bodies: wp.array(dtype=int),
                   body_mass: wp.array2d(dtype=float), dmax: wp.array(dtype=float), k_pair: float, cap_scale: float,
                   timestep: wp.array(dtype=float)):
        w, i = wp.tid()
        r = solref[w, idx[i]]
        tau, zeta, d = r[0], r[1], dmax[i]
        if tau <= 0.0 or zeta <= 0.0:
            return
        wm = w % body_mass.shape[0]
        inv = inv_group(body_mass, wm, side1[i], start, bodies) + inv_group(body_mass, wm, side2[i], start, bodies)
        if inv <= 0.0:
            return
        u = 2.0 * (wp.sqrt(zeta * zeta * d * d + margin * d) - zeta * d)
        cap = cap_scale * u / timestep[w % timestep.shape[0]]
        t = 1.0 / (zeta * wp.sqrt(wp.min(k_pair * (1.0 - d) * inv, cap * cap)))
        if t < tau:
            solref[w, idx[i]] = wp.vec2(t, zeta)

    _KERNEL.append(light_rows)
    return light_rows


def _is_rk4(m: Any) -> bool:
    import mujoco  # noqa: PLC0415

    return int(m.opt.integrator) == int(mujoco.mjtIntegrator.mjINT_RK4)


def _rk4_cap(m: Any, rk4_gains: bool | None) -> bool:
    """Whether the rows take rk4's quarter cap: the entry's gains choice (``GAINS_KEY``, which the MuJoCo entry
    factories pass) when given; else the model's own integrator (direct users, MJCF files that ask for rk4)."""
    return bool(rk4_gains) if rk4_gains is not None else _is_rk4(m)


def _install_warp(solver: Any, rows: list[Rows], rk4: bool) -> None:
    import warp as wp  # noqa: PLC0415

    mw, kernel = solver.mjw_model, _kernel()
    dev = mw.geom_solref.device
    start, bodies = rows[0].members()
    groups = (wp.array(start, dtype=int, device=dev), wp.array(bodies, dtype=int, device=dev))

    def put(a: np.ndarray, dtype: Any) -> Any:
        return wp.array(a, dtype=dtype, device=dev)

    staged = [(r.table, put(r.idx, int), put(r.side1, int), put(r.side2, int), put(r.dmax, float)) for r in rows]
    cap_scale = math.sqrt(RK4_KE_FRACTION) if rk4 else 1.0
    inner = solver._mujoco_warp_step

    def step() -> None:                       # each world's own masses, as Newton last wrote them (notify included)
        for table, idx, side1, side2, dmax in staged:
            arr = getattr(mw, table)
            wp.launch(kernel, dim=(arr.shape[0], idx.shape[0]),
                      inputs=[arr, idx, side1, side2, *groups, mw.body_mass, dmax, K_PAIR, cap_scale,
                              mw.opt.timestep], device=dev)
        inner()

    solver._mujoco_warp_step = step


class _RuledMujoco:
    """Stands in for the `mujoco` module on one MuJoCo C solver: `mj_step` on its model applies the rule first."""

    def __init__(self, module: Any, model: Any, rows: list[Rows], rk4: bool) -> None:
        self._module, self._model, self._rows, self._rk4 = module, model, rows, rk4

    def __getattr__(self, name: str) -> Any:
        return getattr(self._module, name)

    def mj_step(self, model: Any, data: Any, *args: Any) -> None:
        if model is self._model:
            for r in self._rows:
                tab = getattr(model, r.table)
                tab[r.idx] = light_solref(np.asarray(tab[r.idx], dtype=float), r.masses(model.body_mass), r.dmax,
                                          float(model.opt.timestep), self._rk4)
        self._module.mj_step(model, data, *args)


def _install_c(solver: Any, rows: list[Rows], rk4: bool) -> None:
    solver._mujoco = _RuledMujoco(solver._mujoco, solver.mj_model, rows, rk4)


def install_light_body(solver: Any, enabled: bool | None = None, rk4_gains: bool | None = None) -> str:
    """Apply the light-body rule (module docstring) to a built SolverMuJoCo before its first step; returns a summary
    for the route report ("off" / "none" when nothing applies). ``rk4_gains``: the entry's gains choice (F6 R1: an
    implicitfast entry on rk4's gains takes rk4's quarter cap here too); None reads the model's integrator."""
    if enabled is False:
        return "off"
    m = getattr(solver, "mj_model", None)
    if m is None:
        return "none"
    rows, rk4 = candidate_rows(m, file_geoms(solver)), _rk4_cap(m, rk4_gains)
    if not rows:
        return "none"
    if getattr(solver, "use_mujoco_cpu", False):
        _install_c(solver, rows, rk4)
        path = "MuJoCo C"
    elif hasattr(solver, "_mujoco_warp_step"):
        _install_warp(solver, rows, rk4)
        path = "MuJoCo Warp"
    else:
        return "none"
    counts = {r.table: int(r.idx.size) for r in rows}
    log.debug("light-body rule (%s%s): file rows on loose MJCF bodies %s", path, ", rk4" if rk4 else "", counts)
    return f"{path}{', rk4' if rk4 else ''}: {counts}"


def plan(solver: Any, dt: float,
         rk4_gains: bool | None = None) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Census form: {table: (rows, solref before, solref after)} for the rows the rule changes at step `dt` on a
    built SolverMuJoCo's host model (``rk4_gains`` as in ``install_light_body``)."""
    m = solver.mj_model
    out, rk4 = {}, _rk4_cap(m, rk4_gains)
    for r in candidate_rows(m, file_geoms(solver)):
        before = np.asarray(getattr(m, r.table), dtype=float)[r.idx]
        after = light_solref(before, r.mass, r.dmax, dt, rk4)
        ch = np.flatnonzero(np.any(after != before, axis=1))
        out[r.table] = (r.idx[ch], before[ch], after[ch])
    return out


__all__ = ["GAINS_KEY", "GEOM_FILE", "K_PAIR", "LIGHT_KEY", "MJCF_MARK", "candidate_rows", "file_geoms", "geom_files",
           "install_light_body", "light_mass_bound", "light_solref", "loose_bodies", "plan", "weld_masses"]
