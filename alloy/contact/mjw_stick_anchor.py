# SPDX-License-Identifier: Apache-2.0
"""A stick anchor on MuJoCo Warp's elliptic friction rows (WP3, 2026-11-09; known gap rod-chain-grip-slips).

MuJoCo's friction rows are dampers: ``aref_t = -b v_t``, with no position term. A contact that carries a steady load
below its friction limit therefore creeps: with the row's impedance d the solved acceleration is about ``d aref_t +
(1 - d) a0`` (a0: the load's own acceleration at the contact), and it is zero at ``v_t = (1 - d) / d * a0 / b``. A
cable on the chain route, held twisted between a gripper's pads, turned back about 2 % per second on MuJoCo Warp that
way (impratio 30; the chain-grip rule, `alloy.contact.chain_grip`).

The anchor gives the two sliding rows of a covered contact the position term MuJoCo's normal row has:

    aref_t = -k d s_t - b v_t          (the discrete integrator's form of the same law, as constraint._efc_row's)

with ``s`` the slip the pair has made since it touched (the integral of the rows' own velocity, kept tangent to the
contact), and k, b, d the row's own (its solref and solimp). A held load now rests at the offset ``(1 - d) / d * a0 /
k``, well under a micrometre here, instead of creeping.

The anchor sticks and slips. Its pull ``k d |s|`` never passes the pair's friction cone as the rows state it at rest,
``mu aref_n D_n / D_t`` (the normal row's demand times the rows' weights), nor ``cap`` (m/s2). When the slip would ask
for more, the pair is sliding: its slip is dropped and its rows are MuJoCo's own again, bit for bit. It sticks afresh
once MuJoCo's own demand ``b |v_t|`` is back under STICK_RATIO of that limit: a contact sliding at Coulomb's force
asks for more than the cone, a contact creeping under a smaller load asks for less. So a contact that slides keeps
Coulomb's force and gets no extra pull (a pull past the cone would be turned into normal force by the elliptic cone,
the hop of `alloy.contact.mjw_friction_cap`), and a pair that parts loses its anchor.

Covered: contacts between one marked geom (a chain-grip link, `chain_grip.SOLMIX` on its geom) and any other geom, on
a MuJoCo Warp solver that steps once per `make_constraint` (not rk4, whose four stages would count the slip four
times). Up to SLOTS partners per marked geom per world; a further partner keeps MuJoCo's rows.

Every pair takes its slot on the step it first touches, before that step's slip is counted, however many partners a
marked geom meets on that step. The two pads of a grip must be anchored on the same step: with one new partner per
geom per step (the first form), the lower geom id's pad had the anchored rows a step before the other pad, and that
one-sided pull while a falling cable was caught (0.23 m/s on the R4 rig) rolled the cable 20.8 mrad about its axis,
which the anchor then held (the cable end read 1.0725 of the wrist's 0.3 rad instead of 1.0000). A slot whose pair
parts is free from the next step.

Measured, the R4 rig on a clamped chain (tests/api/test_rods_chain_clamp.py; CPU, MuJoCo Warp route, impratio 30;
the cable end's turn / the wrist's): sliding pads 0.9976 at the ramp's end and 0.8651 after a 5 s hold without the
anchor, 1.0000 and 1.0000 with it; pads welded to the wrist 0.9784 and 0.8988 after 0.7 s without, 0.9974 and 0.9976
after 2 s with it. A 0.3 m cube pushed at 0.3 of its friction limit creeps 3.0 um in 2 s without the anchor and
0.0001 um with it; pushed past the limit it slides and coasts the same with and without (0.2723 m in 0.6 s, 0.960
mu g). A pair that slid and is still loaded past STICK_RATIO of its limit does not stick again and creeps as before.
With the anchor on every moving geom the
Coulomb laws read as without it (CPU, quick: A5.onset 0.0 deg, A5.slide 0.45 % / 0.33 %, A5.isotropy 0.093 %,
A5.speed force / mu N 1.000000): tests/contact/test_mjw_stick_anchor.py.

State (``state_arrays``): each slot's partner geom, slip and sliding flag, (worlds, marked geoms, SLOTS). Snapshots,
rollback and resets carry them through `alloy.scene.coordinate_carry.state_arrays`.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import warp as wp

from alloy._vendor.mujoco_warp._src import constraint as _constraint
from alloy._vendor.mujoco_warp._src import types as _types

log = logging.getLogger(__name__)

ANCHOR_KEY = "alloy:stick_anchor"  # MuJoCo entry kwarg: the anchor's cap in m/s2 (None / 0 = off)
CAP_DEFAULT = 20.0                 # m/s2: the anchor's largest pull (module docstring)
ENV = "ALLOY_STICK_ANCHOR"         # "0" turns the anchor off everywhere (A/B runs)
STICK_RATIO = 0.5                  # a sliding pair sticks again when b |v_t| is under this share of its limit
SLOTS = 4                          # partners per marked geom per world
_NONE = 2**30                      # claim: no new partner this step
_MINVAL = 1.0e-15
# fixed point of the per-pair sums (powers of two, exact in float32), each with a clamp far outside what a contact
# row can hold, so that int64 keeps 32 terms at the clamp (a pair has a handful of contacts):
# - a velocity: 2^-44 m/s, up to 1e4 m/s; a normal's component: 2^-40;
# - k d (1/s2): 2^-8, up to 1e15 (a direct solref on a light body is large: a cable link of 1 g on the pair table's
#   2 MN/m has k = 2e9, and the release test must read the pull the rows apply);
# - the cone's pull (m/s2): 2^-24, up to 1e9 (far above any cap); b (1/s): 2^-16, up to 1e12.
_V_SCALE, _V_MAX = 2.0**44, 1.0e4
_N_SCALE = 2.0**40
_K_SCALE, _K_MAX = 2.0**8, 1.0e15
_C_SCALE, _C_MAX = 2.0**24, 1.0e9
_B_SCALE, _B_MAX = 2.0**16, 1.0e12
_REFSAFE = int(_types.DisableBit.REFSAFE)
_vec5 = _types.vec5
_kbimp = _constraint._contact_kbimp


@dataclass
class AnchorState:
    """One solver's anchor: the marked geoms' index, the slots (state) and one step's scratch."""

    cap: float
    index: wp.array        # (ngeom,) the geom's marked index, or -1
    partner: wp.array      # (worlds, marked, SLOTS) the slot's partner geom, or -1          [state]
    slip: wp.array         # (worlds, marked, SLOTS) vec3, the pair's slip in the world frame [state]
    sliding: wp.array      # (worlds, marked, SLOTS) 1 while the pair slides (no anchor), 0 stuck   [state]
    sums: wp.array         # scratch (worlds, marked, SLOTS, 9) int64, fixed point, summed over the pair's contacts:
    #                        sliding velocity [0:3], normal [3:6], k d [6], cone limit mu aref_n D_n / D_t [7], b [8]
    count: wp.array        # scratch: the pair's contacts
    claim: wp.array        # (worlds, marked) scratch: the lowest geom that asks for a slot (one `_ask` round)


@wp.func
def _slot(partner: wp.array3d[int], w: int, i: int, g: int) -> int:
    for j in range(SLOTS):
        if partner[w, i, j] == g:
            return j
    return -1


@wp.func
def _marked(index: wp.array[int], geom: wp.vec2i) -> wp.vec2i:
    """(the marked geom's index, its partner geom), or (-1, -1) unless exactly one of the pair is marked."""
    i0 = index[geom[0]]
    i1 = index[geom[1]]
    if i0 >= 0 and i1 < 0:
        return wp.vec2i(i0, geom[1])
    if i1 >= 0 and i0 < 0:
        return wp.vec2i(i1, geom[0])
    return wp.vec2i(-1, -1)


@wp.func
def _row_gains(disableflags: int, h: float, discrete: int, solref: wp.vec2, solreffriction: wp.vec2, solimp: _vec5,
               pos: float) -> wp.vec3:
    """(k, b, d) of a contact's sliding rows: `constraint._contact_kbimp` on the friction solref (the contact's own
    when solreffriction is zero); in the discrete form k stops at the step's 1 / (h^2 d), as its normal rows do."""
    ref = solref
    if solreffriction[0] != 0.0 or solreffriction[1] != 0.0:
        ref = solreffriction
    kbd = _kbimp(disableflags, h, ref, solimp, pos, discrete != 0)
    k = kbd[0]
    if discrete != 0 and (disableflags & _REFSAFE) == 0:
        k = wp.min(k, 1.0 / wp.max(_MINVAL, h * h * kbd[2]))
    return wp.vec3(k, kbd[1], kbd[2])


@wp.func
def _fixed(x: float, scale: float, limit: float) -> wp.int64:
    """``x`` (clamped to +-``limit``) as a fixed-point integer of ``scale`` counts per unit. The pairs' sums are over
    contacts in whatever order the device runs them: integer sums do not depend on that order, float sums do (D3)."""
    return wp.int64(wp.round(wp.float64(wp.clamp(x, -limit, limit)) * wp.float64(scale)))


@wp.func
def _real(n: wp.int64, scale: float) -> float:
    return float(wp.float64(n) / wp.float64(scale))


@wp.kernel
def _clear(sums: wp.array4d[wp.int64], count: wp.array3d[int], claim: wp.array2d[int]):
    w, i = wp.tid()
    claim[w, i] = _NONE
    for j in range(SLOTS):
        for k in range(9):
            sums[w, i, j, k] = wp.int64(0)
        count[w, i, j] = 0


@wp.kernel
def _ask(nacon: wp.array[int], dim: wp.array[int], world: wp.array[int], geom: wp.array[wp.vec2i],
         addr: wp.array2d[int], etype: wp.array2d[int], index: wp.array[int], partner: wp.array3d[int],
         elliptic: int, claim: wp.array2d[int]):
    """Each covered contact whose pair has no slot asks for one: the lowest partner geom wins this round."""
    c = wp.tid()
    if c >= nacon[0]:
        return
    a0 = addr[c, 0]
    if dim[c] < 3 or a0 < 0 or addr[c, 1] < 0 or addr[c, 2] < 0:     # rows past njmax are dropped one by one (-1)
        return
    w = world[c]
    if etype[w, a0] != elliptic:
        return
    ip = _marked(index, geom[c])
    if ip[0] >= 0 and _slot(partner, w, ip[0], ip[1]) < 0:
        wp.atomic_min(claim, w, ip[0], ip[1])


@wp.kernel
def _grant(claim: wp.array2d[int], partner: wp.array3d[int], slip: wp.array3d[wp.vec3],
           sliding: wp.array3d[int]):
    """The partner that won the round takes the marked geom's first free slot, stuck, its slip at zero."""
    w, i = wp.tid()
    g = claim[w, i]
    claim[w, i] = _NONE
    if g == _NONE:
        return
    for j in range(SLOTS):
        if partner[w, i, j] < 0:
            partner[w, i, j] = g
            slip[w, i, j] = wp.vec3(0.0)
            sliding[w, i, j] = 0
            return


@wp.kernel
def _gather(nacon: wp.array[int], dim: wp.array[int], world: wp.array[int], geom: wp.array[wp.vec2i],
            addr: wp.array2d[int], frame: wp.array[wp.mat33], dist: wp.array[float], margin: wp.array[float],
            solref: wp.array[wp.vec2], solreffriction: wp.array[wp.vec2], solimp: wp.array[_vec5],
            etype: wp.array2d[int], vel: wp.array2d[float], timestep: wp.array[float], index: wp.array[int],
            partner: wp.array3d[int], friction: wp.array[_vec5], efc_d: wp.array2d[float],
            aref: wp.array2d[float], elliptic: int, disableflags: int, discrete: int,
            sums: wp.array4d[wp.int64], count: wp.array3d[int]):
    """Each covered contact adds its sliding velocity, normal, k d, cone limit and b to its pair's slot, as fixed-point
    integers (``_fixed``): the sums are the same in any order."""
    c = wp.tid()
    if c >= nacon[0]:
        return
    a0 = addr[c, 0]
    if dim[c] < 3 or a0 < 0 or addr[c, 1] < 0 or addr[c, 2] < 0:     # rows past njmax are dropped one by one (-1)
        return
    w = world[c]
    if etype[w, a0] != elliptic:
        return
    ip = _marked(index, geom[c])
    if ip[0] < 0:
        return
    j = _slot(partner, w, ip[0], ip[1])
    if j < 0:
        return
    f = frame[c]
    v = vel[w, addr[c, 1]] * f[1] + vel[w, addr[c, 2]] * f[2]
    g = _row_gains(disableflags, timestep[w % timestep.shape[0]], discrete, solref[c], solreffriction[c], solimp[c],
                   dist[c] - margin[c])
    for k in range(3):
        wp.atomic_add(sums, w, ip[0], j, k, _fixed(v[k], _V_SCALE, _V_MAX))
        wp.atomic_add(sums, w, ip[0], j, 3 + k, _fixed(f[0, k], _N_SCALE, 1.0))
    limit = friction[c][0] * wp.max(0.0, aref[w, a0]) * efc_d[w, a0] / wp.max(_MINVAL, efc_d[w, addr[c, 1]])
    wp.atomic_add(sums, w, ip[0], j, 6, _fixed(g[0] * g[2], _K_SCALE, _K_MAX))
    wp.atomic_add(sums, w, ip[0], j, 7, _fixed(limit, _C_SCALE, _C_MAX))
    wp.atomic_add(sums, w, ip[0], j, 8, _fixed(g[1], _B_SCALE, _B_MAX))
    wp.atomic_add(count, w, ip[0], j, 1)


@wp.kernel
def _advance(timestep: wp.array[float], sums: wp.array4d[wp.int64], count: wp.array3d[int], cap: float,
             partner: wp.array3d[int], slip: wp.array3d[wp.vec3], sliding: wp.array3d[int]):
    """Each slot: a stuck pair adds this step's slip (tangent), and starts to slide when its pull would pass the cone
    or the cap; a sliding pair sticks again once MuJoCo's own demand b |v| is well inside that limit; a pair that
    parted gives the slot up."""
    w, i = wp.tid()
    h = timestep[w % timestep.shape[0]]
    for j in range(SLOTS):
        if partner[w, i, j] < 0:
            continue
        n = count[w, i, j]
        if n == 0:
            partner[w, i, j] = -1
            slip[w, i, j] = wp.vec3(0.0)
            continue
        inv = 1.0 / float(n)
        nrm = wp.normalize(wp.vec3(_real(sums[w, i, j, 3], _N_SCALE), _real(sums[w, i, j, 4], _N_SCALE),
                                   _real(sums[w, i, j, 5], _N_SCALE)))
        v = inv * wp.vec3(_real(sums[w, i, j, 0], _V_SCALE), _real(sums[w, i, j, 1], _V_SCALE),
                          _real(sums[w, i, j, 2], _V_SCALE))
        v = v - nrm * wp.dot(v, nrm)
        pull = wp.min(cap, _real(sums[w, i, j, 7], _C_SCALE) * inv)
        if sliding[w, i, j] != 0:
            if _real(sums[w, i, j, 8], _B_SCALE) * inv * wp.length(v) < STICK_RATIO * pull:
                sliding[w, i, j] = 0
            continue
        s = slip[w, i, j] + h * v
        s = s - nrm * wp.dot(s, nrm)
        if wp.length(s) * _real(sums[w, i, j, 6], _K_SCALE) * inv > pull:
            sliding[w, i, j] = 1
            s = wp.vec3(0.0)
        slip[w, i, j] = s


@wp.kernel
def _apply(nacon: wp.array[int], dim: wp.array[int], world: wp.array[int], geom: wp.array[wp.vec2i],
           addr: wp.array2d[int], frame: wp.array[wp.mat33], dist: wp.array[float], margin: wp.array[float],
           solref: wp.array[wp.vec2], solreffriction: wp.array[wp.vec2], solimp: wp.array[_vec5],
           etype: wp.array2d[int], vel: wp.array2d[float], timestep: wp.array[float], index: wp.array[int],
           partner: wp.array3d[int], slip: wp.array3d[wp.vec3], sliding: wp.array3d[int], elliptic: int,
           disableflags: int, discrete: int, aref: wp.array2d[float]):
    """The two sliding rows of each covered contact of a stuck pair: the row's law with the pair's slip as its
    position. A sliding pair keeps MuJoCo's rows."""
    c = wp.tid()
    if c >= nacon[0]:
        return
    a0 = addr[c, 0]
    if dim[c] < 3 or a0 < 0 or addr[c, 1] < 0 or addr[c, 2] < 0:     # rows past njmax are dropped one by one (-1)
        return
    w = world[c]
    if etype[w, a0] != elliptic:
        return
    ip = _marked(index, geom[c])
    if ip[0] < 0:
        return
    j = _slot(partner, w, ip[0], ip[1])
    if j < 0:
        return
    if sliding[w, ip[0], j] != 0:
        return
    h = timestep[w % timestep.shape[0]]
    g = _row_gains(disableflags, h, discrete, solref[c], solreffriction[c], solimp[c], dist[c] - margin[c])
    kd = g[0] * g[2]
    s = slip[w, ip[0], j]
    f = frame[c]
    for r in range(1, 3):
        e = addr[c, r]
        p = wp.dot(s, f[r])
        v = vel[w, e]
        if discrete != 0:
            aref[w, e] = (-kd * p - (g[1] + h * kd) * v) / (1.0 + h * g[1] + h * h * kd)
        else:
            aref[w, e] = -kd * p - g[1] * v


def launch_anchor(st: AnchorState, m: Any, d: Any) -> None:
    """One step of the anchor on MuJoCo Warp model / data ``m`` / ``d``, after their ``make_constraint``."""
    elliptic = int(_types.ConstraintType.CONTACT_ELLIPTIC)
    flags = int(m.opt.disableflags)
    discrete = int(int(m.opt.integrator) == int(_types.IntegratorType.DISCRETE))
    con = d.contact
    rows = [d.nacon, con.dim, con.worldid, con.geom, con.efc_address, con.frame, con.dist, con.includemargin,
            con.solref, con.solreffriction, con.solimp, d.efc.type, d.efc.vel, m.opt.timestep, st.index, st.partner]
    scratch = [st.sums, st.count]
    wp.launch(_clear, dim=st.claim.shape, inputs=[*scratch, st.claim])
    for _ in range(SLOTS):   # one new partner per marked geom per round: SLOTS rounds seat all that can be seated
        wp.launch(_ask, dim=d.naconmax, inputs=[d.nacon, con.dim, con.worldid, con.geom, con.efc_address,
                                                d.efc.type, st.index, st.partner, elliptic, st.claim])
        wp.launch(_grant, dim=st.claim.shape, inputs=[st.claim, st.partner, st.slip, st.sliding])
    wp.launch(_gather, dim=d.naconmax, inputs=[*rows, con.friction, d.efc.D, d.efc.aref, elliptic, flags, discrete,
                                               *scratch])
    wp.launch(_advance, dim=st.claim.shape, inputs=[m.opt.timestep, *scratch, float(st.cap), st.partner, st.slip,
                                                    st.sliding])
    wp.launch(_apply, dim=d.naconmax, inputs=[*rows, st.slip, st.sliding, elliptic, flags, discrete, d.efc.aref])


def anchored_make_constraint(inner: Any, st: AnchorState) -> Any:
    """``inner`` (MuJoCo Warp's ``make_constraint``) followed by the anchor."""

    def make_constraint(m: Any, d: Any) -> None:
        inner(m, d)
        launch_anchor(st, m, d)

    make_constraint._alloy_stick_anchor = st.cap
    return make_constraint


def cap_value(request: Any) -> float | None:
    """The cap an entry asks for (``ANCHOR_KEY``'s value), or None when off (None, 0, ``ALLOY_STICK_ANCHOR=0``)."""
    if request is None or os.environ.get(ENV, "1") == "0":
        return None
    cap = float(request)
    if cap < 0.0:
        raise ValueError(f"{ANCHOR_KEY} must be a cap in m/s2 (>= 0, 0 = off), got {cap}")
    return cap or None


def make_state(geoms: Any, ngeom: int, nworld: int, cap: float, device: Any) -> AnchorState:
    """The anchor's arrays for the marked ``geoms`` (MuJoCo geom ids) of a model with ``ngeom`` geoms."""
    geoms = np.unique(np.asarray(geoms, dtype=np.int32))
    index = np.full(ngeom, -1, dtype=np.int32)
    index[geoms] = np.arange(len(geoms), dtype=np.int32)
    shape = (nworld, len(geoms), SLOTS)
    return AnchorState(
        cap=float(cap), index=wp.array(index, dtype=int, device=device),
        partner=wp.full(shape, -1, dtype=int, device=device), slip=wp.zeros(shape, dtype=wp.vec3, device=device),
        sliding=wp.zeros(shape, dtype=int, device=device),
        sums=wp.zeros((*shape, 9), dtype=wp.int64, device=device),
        count=wp.zeros(shape, dtype=int, device=device), claim=wp.full(shape[:2], _NONE, dtype=int, device=device))


def install_stick_anchor(solver: Any, geoms: Any, request: Any = CAP_DEFAULT) -> bool:
    """Anchor the friction rows of this solver's contacts with the marked ``geoms`` (MuJoCo geom ids), at
    ``cap_value(request)`` m/s2. Call before the first step or graph capture. False when the anchor is off, no geom
    is marked, or the solver has no once-per-step MuJoCo Warp constraint build (MuJoCo C, rk4)."""
    cap = cap_value(request)
    geoms = np.asarray(geoms, dtype=np.int32).ravel()
    orig = getattr(solver, "_mujoco_warp_step", None)
    if cap is None or geoms.size == 0 or orig is None or getattr(solver, "use_mujoco_cpu", False):
        return False
    if getattr(solver, "_alloy_stick_anchor", None) is not None:
        raise RuntimeError("this solver's friction rows are already anchored")
    m, d = solver.mjw_model, solver.mjw_data
    if int(m.opt.integrator) == int(_types.IntegratorType.RK4):
        log.warning("mjw_stick_anchor: rk4 builds its constraints four times a step, so this entry keeps MuJoCo's "
                    "friction rows (a held contact creeps: known gap rod-chain-grip-slips)")
        return False
    st = make_state(geoms, int(m.ngeom), int(d.nworld), cap, d.nacon.device)

    def step(*args: Any, **kwargs: Any) -> Any:
        prev = _constraint.make_constraint
        _constraint.make_constraint = anchored_make_constraint(prev, st)
        try:
            return orig(*args, **kwargs)
        finally:
            _constraint.make_constraint = prev

    solver._mujoco_warp_step = step
    solver._alloy_stick_anchor = st
    log.info("mjw_stick_anchor: %d geoms' friction rows anchored (cap %g m/s2, %d slots each)", geoms.size, cap,
             SLOTS)
    return True


def state_arrays(solver: Any) -> dict[str, Any]:
    """The anchor's solver state, (worlds, marked geoms, SLOTS) each: restored with the solver's other state."""
    st = getattr(solver, "_alloy_stick_anchor", None)
    if st is None:
        return {}
    return {"stick_anchor.partner": st.partner, "stick_anchor.slip": st.slip, "stick_anchor.sliding": st.sliding}


__all__ = ["ANCHOR_KEY", "CAP_DEFAULT", "ENV", "SLOTS", "STICK_RATIO", "AnchorState", "anchored_make_constraint",
           "cap_value", "install_stick_anchor", "launch_anchor", "make_state", "state_arrays"]
