# SPDX-License-Identifier: Apache-2.0
"""MixedContactBinding: append the layer's precision rows into SolverMuJoCo through MJWarp ``contactfilter`` (D10).

Spike result (docs/notes/contact-append-spike.md): rows written after MJWarp's native contacts inside
``contactfilter`` (mujoco-warp 3.16.0 ``_src/collision_driver.py:987``, before ``make_constraint``) behave like the
same rows on the ``native="newton"`` path. This kernel replaces Newton's converter
(newton/_src/solvers/mujoco/kernels.py:403-700) for the appended range:

- rows go to ``nacon + i`` with no atomics, so their order is the layer's (deterministic) order;
- it writes ``type = CONSTRAINT``, ``adhesion = 0``, ``geomcollisionid = -1``, which Newton's ``write_contact`` leaves
  stale (MJWarp builds rows only for ``type & CONSTRAINT``, constraint.py:2803);
- ``includemargin`` is the Newton shape margins (MJWarp may zero ``geom_margin`` for MULTICCD);
- solref comes from the layer's per-row K (D22, pair K x weight or hydroelastic area·p/|depth|) with the pair's flat
  solimp ``imp = d0 = dmax``: ``timeconst = 1/sqrt(K·invweight·(1-imp))`` (kernels.py:208-222, 663-685), so the
  realised stiffness is ``K`` until MJWarp's ``τ ≥ 2·dt`` clamp (``k_real_mujoco``);
- friction is MuJoCo's geom mix (priority, else max) scaled by ``rigid_contact_friction`` (D13).

A device flag armed by ``AlloyContactLayer.collide`` and cleared here keeps the second ``collision()`` pass under
sleeping (forward.py:1449-1460) from appending twice.
"""

from __future__ import annotations

import warp as wp

from alloy._vendor import newton
from alloy.contact.speculative import l1_callback_available, prepare_speculative_row, set_speculative_targets, vec5
from alloy.contact.types import AlloyContacts
from alloy.core.errors import PrecisionSpecError
from alloy.core.log import get_logger

log = get_logger("contact.mujoco_append")

MJ_MINVAL = 1.0e-15
MJ_MINMU = 1.0e-5
CONTACT_CONSTRAINT = 1          # mujoco_warp ContactType.CONSTRAINT
INACTIVE_DIST = 1.0e6           # skipped rows: never active
SUPPORTED_INTEGRATORS = {0: "euler", 3: "implicitfast", 4: "discrete"}   # mjtIntegrator values


@wp.struct
class AppendRows:
    count: wp.array(dtype=wp.int32)
    shape0: wp.array(dtype=wp.int32)
    shape1: wp.array(dtype=wp.int32)
    point0: wp.array(dtype=wp.vec3)
    point1: wp.array(dtype=wp.vec3)
    offset0: wp.array(dtype=wp.vec3)
    offset1: wp.array(dtype=wp.vec3)
    normal: wp.array(dtype=wp.vec3)
    margin0: wp.array(dtype=wp.float32)
    margin1: wp.array(dtype=wp.float32)
    friction_scale: wp.array(dtype=wp.float32)
    row_k: wp.array(dtype=wp.float32)
    pair: wp.array(dtype=wp.int32)
    world: wp.array(dtype=wp.int32)
    pair_imp: wp.array(dtype=wp.float32)
    pair_l1: wp.array(dtype=wp.int32)
    body_q: wp.array(dtype=wp.transform)
    shape_body: wp.array(dtype=wp.int32)
    body_flags: wp.array(dtype=wp.int32)
    shape_margin: wp.array(dtype=wp.float32)


@wp.struct
class MjGeom:
    s2g: wp.array(dtype=wp.int32)
    geom_bodyid: wp.array(dtype=int)
    body_weldid: wp.array(dtype=int)
    body_dofnum: wp.array(dtype=int)
    body_invweight0: wp.array2d(dtype=wp.vec2)
    geom_condim: wp.array(dtype=int)
    geom_priority: wp.array(dtype=int)
    geom_friction: wp.array2d(dtype=wp.vec3)
    geom_solref: wp.array2d(dtype=wp.vec2)
    geom_solimp: wp.array2d(dtype=vec5)
    impratio_invsqrt: wp.array(dtype=float)
    timestep: wp.array(dtype=float)
    elliptic: int


@wp.struct
class MjContactOut:
    dist: wp.array(dtype=float)
    pos: wp.array(dtype=wp.vec3)
    frame: wp.array(dtype=wp.mat33)
    includemargin: wp.array(dtype=float)
    friction: wp.array(dtype=vec5)
    solref: wp.array(dtype=wp.vec2)
    solreffriction: wp.array(dtype=wp.vec2)
    solimp: wp.array(dtype=vec5)
    dim: wp.array(dtype=int)
    geom: wp.array(dtype=wp.vec2i)
    efc_address: wp.array2d(dtype=int)
    worldid: wp.array(dtype=int)
    ctype: wp.array(dtype=int)
    adhesion: wp.array(dtype=float)
    geomcollisionid: wp.array(dtype=int)
    clearance: wp.array(dtype=float)


@wp.func
def _frame(n: wp.vec3) -> wp.mat33:
    """MuJoCo contact frame: rows normal, t1, t2 (mujoco_warp math.make_frame / orthogonals)."""
    a = wp.normalize(n)
    b = wp.vec3(0.0, 0.0, 1.0)
    if -0.5 < a[1] and a[1] < 0.5:
        b = wp.vec3(0.0, 1.0, 0.0)
    b = wp.normalize(b - a * wp.dot(a, b))
    c = wp.cross(a, b)
    return wp.mat33(a[0], a[1], a[2], b[0], b[1], b[2], c[0], c[1], c[2])


@wp.func
def _immovable(body: int, mj_body: int, g: MjGeom, rows: AppendRows) -> bool:
    if body < 0:
        return True
    if (rows.body_flags[body] & int(newton.BodyFlags.KINEMATIC)) != 0:
        return True
    return g.body_dofnum[g.body_weldid[mj_body]] == 0


@wp.func
def _mix(ga: int, gb: int, w: int, g: MjGeom) -> tuple[int, vec5]:
    """MuJoCo condim/friction mix: higher priority wins, else max (collision_core.contact_params)."""
    fa = g.geom_friction[w % g.geom_friction.shape[0], ga]
    fb = g.geom_friction[w % g.geom_friction.shape[0], gb]
    condim = wp.max(g.geom_condim[ga], g.geom_condim[gb])
    f = wp.max(fa, fb)
    if g.geom_priority[ga] > g.geom_priority[gb]:
        condim = g.geom_condim[ga]
        f = fa
    elif g.geom_priority[gb] > g.geom_priority[ga]:
        condim = g.geom_condim[gb]
        f = fb
    fr = vec5(wp.max(MJ_MINMU, f[0]), wp.max(MJ_MINMU, f[0]), wp.max(MJ_MINMU, f[1]), wp.max(MJ_MINMU, f[2]),
              wp.max(MJ_MINMU, f[2]))
    return condim, fr


@wp.func
def _solref(k: float, invweight: float, imp: float, mu: float, condim: int, ir: float, elliptic: int) -> wp.vec2:
    """Force-space K -> MuJoCo (timeconst, dampratio=1) with the resting-contact gain factor (kernels.py:208-222)."""
    factor = invweight * (1.0 - imp)
    if elliptic == 0 and condim > 1:
        mu2 = mu * mu
        factor = factor * mu2 * (1.0 + mu2) * ir * ir / float(condim - 1)
    ke = wp.max(k * factor, MJ_MINVAL)
    return wp.vec2(wp.sqrt(1.0 / ke), 1.0)


@wp.func
def _write_inactive(cid: int, out: MjContactOut):
    out.dist[cid] = INACTIVE_DIST
    out.includemargin[cid] = 0.0
    out.ctype[cid] = 0
    out.adhesion[cid] = 0.0
    out.geomcollisionid[cid] = -1
    out.clearance[cid] = 0.0
    out.worldid[cid] = 0
    for j in range(out.efc_address.shape[1]):
        out.efc_address[cid, j] = -1


@wp.func
def _point(body: int, p: wp.vec3, o: wp.vec3, rows: AppendRows) -> wp.vec3:
    if body < 0:
        return p + o
    return wp.transform_point(rows.body_q[body], p + o)


@wp.kernel
def append_rows_kernel(pending: wp.array[wp.int32], start: wp.array[wp.int32], naconmax: int, rows: AppendRows,
                       g: MjGeom, out: MjContactOut):
    i = wp.tid()
    if pending[0] == 0 or i >= rows.count[0]:
        return
    cid = start[0] + i
    if cid >= naconmax:
        return
    sa = rows.shape0[i]
    sb = rows.shape1[i]
    ga = g.s2g[sa]
    gb = g.s2g[sb]
    ba = rows.shape_body[sa]
    bb = rows.shape_body[sb]
    if ga < 0 or gb < 0:
        _write_inactive(cid, out)
        return
    mja = g.geom_bodyid[ga]
    mjb = g.geom_bodyid[gb]
    if _immovable(ba, mja, g, rows) and _immovable(bb, mjb, g, rows):
        _write_inactive(cid, out)
        return
    w = rows.world[i]
    ra = rows.margin0[i] - rows.shape_margin[sa]          # effective radius without Newton's artificial margin
    rb = rows.margin1[i] - rows.shape_margin[sb]
    n = rows.normal[i]
    xa = _point(ba, rows.point0[i], wp.vec3(0.0), rows)
    xb = _point(bb, rows.point1[i], wp.vec3(0.0), rows)
    sxa = _point(ba, rows.point0[i], rows.offset0[i] * wp.where(rows.margin0[i] != 0.0, ra / rows.margin0[i], 0.0),
                 rows)
    sxb = _point(bb, rows.point1[i], rows.offset1[i] * wp.where(rows.margin1[i] != 0.0, rb / rows.margin1[i], 0.0),
                 rows)
    condim, fr = _mix(ga, gb, w, g)
    scale = rows.friction_scale[i]
    if scale > 0.0:
        fr = vec5(fr[0] * scale, fr[1] * scale, fr[2], fr[3], fr[4])
        if g.elliptic == 0 and condim > 1:
            fr = vec5(wp.max(MJ_MINMU, fr[0]), wp.max(MJ_MINMU, fr[1]), fr[2], fr[3], fr[4])
    p = rows.pair[i]
    imp = rows.pair_imp[wp.max(p, 0)]
    invw = g.body_invweight0[w % g.body_invweight0.shape[0], mja][0] + \
        g.body_invweight0[w % g.body_invweight0.shape[0], mjb][0]
    solref = g.geom_solref[w % g.geom_solref.shape[0], ga]
    solimp = g.geom_solimp[w % g.geom_solimp.shape[0], ga]
    k = rows.row_k[i]
    if k > 0.0 and invw > 0.0:
        solref = _solref(k, invw, imp, fr[0], condim, g.impratio_invsqrt[w % g.impratio_invsqrt.shape[0]], g.elliptic)
        solimp = vec5(imp, imp, 0.001, 1.0, 0.5)
    out.dist[cid] = wp.dot(n, xb - xa) - (ra + rb)
    out.pos[cid] = 0.5 * (sxa + sxb)
    out.frame[cid] = _frame(n)
    out.includemargin[cid] = rows.shape_margin[sa] + rows.shape_margin[sb]
    out.friction[cid] = fr
    out.solref[cid] = solref
    out.solreffriction[cid] = wp.vec2(0.0, 0.0)
    out.solimp[cid] = solimp
    out.dim[cid] = condim
    out.geom[cid] = wp.vec2i(ga, gb)
    out.worldid[cid] = w
    out.ctype[cid] = CONTACT_CONSTRAINT
    out.adhesion[cid] = 0.0
    out.geomcollisionid[cid] = -1
    for j in range(out.efc_address.shape[1]):
        out.efc_address[cid, j] = -1
    out.clearance[cid] = 0.0
    if p >= 0 and rows.pair_l1[p] != 0:
        prepare_speculative_row(cid, g.timestep[w % g.timestep.shape[0]], out.dist, out.includemargin, out.dim,
                                out.solref, out.solimp, out.clearance)


@wp.kernel
def _guard(pending: wp.array[wp.int32], nacon: wp.array[int], count: wp.array[wp.int32], naconmax: int,
           start: wp.array[wp.int32], end: wp.array[wp.int32], overflow: wp.array[wp.int32]):
    """Fix the append range ``[nacon, nacon + count)`` and the new ``nacon`` once per armed step."""
    if pending[0] == 0:
        return
    s = nacon[0]
    e = s + count[0]
    if e > naconmax:
        wp.atomic_add(overflow, 0, e - naconmax)
        e = naconmax
    start[0] = s
    end[0] = e
    nacon[0] = e


@wp.kernel
def _set_flag(flag: wp.array[wp.int32], value: int):
    flag[0] = value


@wp.kernel
def _range_ids(start: wp.array[wp.int32], ids: wp.array[int]):
    i = wp.tid()
    ids[i] = start[0] + i


@wp.kernel
def _store_forces(count: wp.array[wp.int32], start: wp.array[wp.int32], end: wp.array[wp.int32],
                  f: wp.array[wp.spatial_vector], out: wp.array[wp.vec3]):
    i = wp.tid()
    if i < count[0] and start[0] + i < end[0]:
        out[i] = wp.spatial_top(f[i])          # force on geom2 = shape1 (B) from A, world frame (SPEC §4.1)
    elif i < out.shape[0]:
        out[i] = wp.vec3(0.0)


class MixedContactBinding:
    """Installs the precision append on a ``SolverMuJoCo(use_mujoco_contacts=True)``; see the module docstring."""

    def __init__(self, solver: newton.solvers.SolverMuJoCo, layer) -> None:
        self.solver, self.layer = solver, layer
        self.device = layer.device
        self.pending = wp.zeros(1, dtype=wp.int32, device=self.device)
        self.start = wp.zeros(1, dtype=wp.int32, device=self.device)
        self.end = wp.zeros(1, dtype=wp.int32, device=self.device)
        self.overflow = wp.zeros(1, dtype=wp.int32, device=self.device)
        self.l1 = any(s.ccd != "off" for s in layer.pairs)
        self._installed = False

    @property
    def installed(self) -> bool:
        return self._installed

    # ------------------------------------------------------------------------------------------- checks ---
    def _integrator(self) -> str:
        code = int(self.solver.mjw_model.opt.integrator)
        if code not in SUPPORTED_INTEGRATORS:
            raise PrecisionSpecError(f"precision append does not support MuJoCo integrator {code} (rk4/implicit re-run "
                                     f"collision at stale contacts); use euler, implicitfast or discrete")
        return SUPPORTED_INTEGRATORS[code]

    def _check(self) -> None:
        if self.solver.use_mujoco_cpu or not bool(self.solver.mjw_model.opt.run_collision_detection):
            raise PrecisionSpecError("MixedContactBinding needs SolverMuJoCo(use_mujoco_contacts=True) on MuJoCo Warp")
        integ = self._integrator()
        if integ == "discrete" and self.l1:
            raise PrecisionSpecError("discrete integrator cannot carry L1 speculative rows (D7, §3.5 table): set "
                                     "ccd='off' on the pair or use euler/implicitfast")
        if self.l1 and not l1_callback_available(self.solver.mjw_model.opt.disableflags):
            raise PrecisionSpecError("L1 needs MJWarp's passive callback, skipped when SPRING and DAMPER are disabled")
        cb = self.solver.mjw_model.callback
        for name in ("contactfilter", "passive"):
            current = getattr(cb, name)
            if current is not None and getattr(current, "__self__", None) is not self:
                raise PrecisionSpecError(f"mjw_model.callback.{name} is already set by someone else")

    def install(self) -> None:
        """Set ``contactfilter`` (and ``passive`` for L1). Raises for rk4/implicit, for discrete with L1."""
        self._check()
        self._build_args()
        cb = self.solver.mjw_model.callback
        cb.contactfilter = self._contactfilter
        if self.l1:
            cb.passive = self._passive
        self._installed = True
        log.info("precision append installed: %d pairs, L1=%s, integrator=%s", len(self.layer.pairs), self.l1,
                 self._integrator())

    # --------------------------------------------------------------------------------------------- args ---
    def _build_args(self) -> None:
        import numpy as np  # noqa: PLC0415

        s, m, d, lay = self.solver, self.solver.mjw_model, self.solver.mjw_data, self.layer
        model = lay.model
        g2s = s.mjc_geom_to_newton_shape.numpy()
        s2g = np.full(model.shape_count, -1, np.int32)
        for w in range(g2s.shape[0]):
            ok = g2s[w] >= 0
            s2g[g2s[w][ok]] = np.nonzero(ok)[0]
        g = MjGeom()
        g.s2g = wp.array(s2g, dtype=wp.int32, device=self.device)
        for name in ("geom_bodyid", "body_weldid", "body_dofnum", "body_invweight0", "geom_condim", "geom_priority",
                     "geom_friction", "geom_solref", "geom_solimp"):
            setattr(g, name, getattr(m, name))
        g.impratio_invsqrt, g.timestep = m.opt.impratio_invsqrt, m.opt.timestep
        g.elliptic = int(int(m.opt.cone) == 1)
        self._geom = g
        self._out = MjContactOut()
        for name in ("dist", "pos", "frame", "includemargin", "friction", "solref", "solreffriction", "solimp", "dim",
                     "geom", "efc_address", "worldid", "adhesion", "geomcollisionid"):
            setattr(self._out, name, getattr(d.contact, name))
        self._out.ctype = d.contact.type
        self._out.clearance = wp.zeros(d.naconmax, dtype=float, device=self.device)
        self._rows = self._make_rows(lay, model)
        self._ids = wp.zeros(lay.contacts().rigid_contact_max, dtype=int, device=self.device)
        self._forces = wp.zeros(lay.contacts().rigid_contact_max, dtype=wp.spatial_vector, device=self.device)

    def _make_rows(self, lay, model: newton.Model) -> AppendRows:
        import numpy as np  # noqa: PLC0415

        c, side = lay.contacts(), lay.sidecar()
        r = AppendRows()
        for name in ("shape0", "shape1", "point0", "point1", "offset0", "offset1", "normal", "margin0", "margin1"):
            setattr(r, name, getattr(c, f"rigid_contact_{name}"))
        r.count, r.friction_scale, r.row_k = c.rigid_contact_count, c.rigid_contact_friction, lay.row_stiffness
        r.pair, r.world, r.pair_imp = side.pair, side.world, lay.pair_imp
        r.pair_l1 = wp.array(np.asarray([int(p.ccd != "off") for p in lay.pairs] or [0], np.int32), dtype=wp.int32,
                             device=self.device)
        r.body_q = model.body_q
        r.shape_body, r.body_flags, r.shape_margin = model.shape_body, model.body_flags, model.shape_margin
        return r

    # ----------------------------------------------------------------------------------------- runtime ---
    def arm(self, state: newton.State) -> None:
        """Called by ``AlloyContactLayer.collide``: the next ``collision()`` appends the rows posed at ``state``."""
        if not self._installed:
            return
        self._rows.body_q = state.body_q
        wp.launch(_set_flag, dim=1, inputs=[self.pending, 1], device=self.device)

    def _contactfilter(self, m, d) -> None:
        wp.launch(_guard, dim=1, device=self.device, inputs=[self.pending, d.nacon, self._rows.count, d.naconmax,
                                                             self.start, self.end, self.overflow])
        wp.launch(append_rows_kernel, dim=self.layer.contacts().rigid_contact_max, device=self.device,
                  inputs=[self.pending, self.start, d.naconmax, self._rows, self._geom, self._out])
        wp.launch(_set_flag, dim=1, inputs=[self.pending, 0], device=self.device)

    def _passive(self, m, d) -> None:
        c = d.contact
        wp.launch(set_speculative_targets, dim=d.naconmax, device=self.device,
                  inputs=[self.start, self.end, m.opt.timestep, c.worldid, self._out.clearance, c.efc_address, c.dist,
                          c.includemargin, d.efc.vel, d.efc.pos, d.efc.aref])

    def precision_forces(self, contacts: AlloyContacts) -> None:
        """MuJoCo constraint forces of the appended rows -> ``contacts.rigid_contact_force`` (world frame, on B)."""
        from alloy._vendor import mujoco_warp  # noqa: PLC0415

        n = contacts.rigid_contact_max
        wp.launch(_range_ids, dim=n, inputs=[self.start, self._ids], device=self.device)
        self._forces.zero_()
        mujoco_warp.contact_force(self.solver.mjw_model, self.solver.mjw_data, self._ids, True, self._forces)
        wp.launch(_store_forces, dim=n, device=self.device,
                  inputs=[contacts.rigid_contact_count, self.start, self.end, self._forces, contacts.rigid_contact_force])
