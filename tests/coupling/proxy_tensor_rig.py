# SPDX-License-Identifier: Apache-2.0
"""Rig of the proxy tensor law (``test_proxy_tensor_law.py``; known gap coupling-proxy-isotropic-mass).

C3's 0.515 kg platen (a 6 x 6 x 2 cm box) above its free 16 g silicone pad, mid-air, from the public API. Every kind
is the same MJCF body and differs only in its joint, so shapes, contact settings and the exchange clock are the same:
``slider`` (one vertical slide: W = diag(0, 0, 1/m), proxy 3 m), ``xyz`` (three slides: W = I/m, proxy exact, no
rotation, the slider's twin), ``free`` (a free joint: exact) and ``hinge`` (a hinge 5 cm from the centre of mass, which
then moves along z: proxy about 3x its mass along z).

Two taps, both on the built simulation's coupler and neither changing a result it keeps:
- ``ExchangeTap`` records every proxy pass: the feedback force the pass used, the proxy's velocity in VBD before its
  step (the source's stepped velocity) and after it.
- ``inject`` writes a known feedback force before each exchange. Pass 0 applies it to the MuJoCo body and takes it back
  from the proxy inside VBD (the lagged rewind); pass 1 restarts the exchange with VBD's own harvest (zero in free
  flight). The two passes differ by that force alone: the source's velocity by dt W f, the proxy's VBD step by
  -dt Minv f. A proxy that answers as its body does leaves nothing.

``contact`` starts the platen inside the pad's reach (``TOUCH``): exchange 0 is then the same contact from the same
state for every kind, and its first pass has no feedback yet, so the proxy's velocity change in it is VBD's contact
solve alone. Later exchanges drift apart with the float32 grid the scene sits on (``build(origin=)``).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import numpy as np

import alloy

M_PLATEN = 0.5153                 # kg (C3's platen)
HALF = (0.03, 0.03, 0.01)         # platen half sizes (m)
PAD = (0.04, 0.04, 0.01)          # pad full sizes (m): 16 g of tissue_fat_silicone
RADIUS = 0.000571                 # the pad's particle radius (m)
GAP = 3 * RADIUS                  # platen bottom to pad top at the start (m): free flight for 11 ms at 0.1 m/s
TOUCH = 0.8 * RADIUS              # ... of the contact rig: the platen starts 0.2 radii inside the pad's reach
ARM = 0.05                        # hinge axis to the platen's centre of mass (m)
# The hinged platen's mass along z at its centre of mass, 1 / W_zz = m + I_yy / ARM^2 (0.584 kg).
HINGE_AXIS_MASS = M_PLATEN * (1.0 + (HALF[0] ** 2 + HALF[2] ** 2) / (3.0 * ARM ** 2))
DT = 0.0025                       # control period (s)
REF_DROP = 0.2                    # world 1's pad starts this much lower: the free-fall reference (m)

JOINTS: dict[str, str] = {
    "slider": '<joint name="z" type="slide" axis="0 0 1"/>',
    "xyz": '<joint name="x" type="slide" axis="1 0 0"/><joint name="y" type="slide" axis="0 1 0"/>'
           '<joint name="z" type="slide" axis="0 0 1"/>',
    "free": "<freejoint/>",
    "hinge": f'<joint name="h" type="hinge" axis="0 1 0" pos="{ARM} 0 0"/>',
}


def platen_mjcf(kind: str, mass: float = M_PLATEN, geom: str | None = None) -> str:
    """One body named ``platen`` on the joint(s) of ``kind``; ``geom`` replaces the box (attributes of a geom)."""
    shape = geom or f'type="box" size="{HALF[0]} {HALF[1]} {HALF[2]}"'
    return (f'<mujoco><worldbody><body name="platen" pos="0 0 0">{JOINTS[kind]}'
            f'<geom {shape} mass="{mass}"/></body></worldbody></mujoco>')


@dataclass
class Rig:
    """A built platen-over-pad simulation and the handles the law reads."""

    kind: str
    sim: Any                       # alloy Simulation (public API)
    sched: Any                     # the coupler (``sim.engine.stepper``)
    vbd: Any                       # the VBD entry's solver
    mapping: Any                   # the MuJoCo -> VBD body proxy mapping
    bodies: np.ndarray             # the platen's model body id per world
    local: int                     # world 0's platen in the VBD entry (its proxy)
    mass: float

    @property
    def body(self) -> int:
        return int(self.bodies[0])

    @property
    def proxy_mass(self) -> float:
        """The scalar mass installed for world 0's proxy in VBD (kg)."""
        return float(self.vbd.model.body_mass.numpy()[self.local])


def build(kind: str, *, mass: float = M_PLATEN, geom: str | None = None, worlds: int = 1,
          half_height: float = HALF[2], origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
          gap: float = GAP) -> Rig:
    """The platen of ``kind`` with its centre of mass at ``origin``, the pad ``gap`` below it. With two worlds, world
    1's pad starts ``REF_DROP`` lower and is never touched (the free-fall reference of ``contact``). The origin only
    moves the float32 grid the scene sits on (finest at 0): the law's resolution floor is the same scene elsewhere."""
    scene = alloy.Scene(worlds=worlds, dt=DT)
    x, y, z = origin[0], origin[1], origin[2] - (half_height + gap + PAD[2] / 2)
    pos = np.array([[x, y, z - REF_DROP * w] for w in range(worlds)])
    scene.add_box("pad", size=PAD, pos=pos if worlds > 1 else tuple(pos[0]), material="tissue_fat_silicone")
    scene.add_robot("rig", mjcf=platen_mjcf(kind, mass, geom), pos=origin)
    sim = alloy.build(scene, device="cpu")
    eng = sim.engine
    sched = eng.stepper
    vbd = eng.parts.solvers["vbd"]
    bodies = np.asarray(sim.index.per_world(sim.index.body("rig/platen"), "body")).ravel()
    mapping = getattr(sched, "_proxy_mappings", [None])[0]
    local = -1
    if mapping is not None:
        src = mapping.src_ids.numpy()
        local = int(mapping.proxy_ids_local.numpy()[int(np.flatnonzero(src == bodies[0])[0])])
    return Rig(kind, sim, sched, vbd, mapping, bodies, local, mass)


def coupling_class(rig: Rig) -> dict[str, Any]:
    """Which coupler runs the pair and whether it is a strong pair (D5), for the notes and the hinge-tip check."""
    sets = getattr(rig.sched, "_sets", [])
    return {"coupler": type(rig.sched).__name__, "routes": {k: r["route"] for k, r in rig.sim.routes().items()},
            "strong": [bool(es.strong) for es in sets], "staggered": [bool(es.staggered) for es in sets],
            "has_proxy": rig.local >= 0}


def launch(rig: Rig, v0: float) -> None:
    """Give the platen a downward speed ``v0`` at its centre of mass in every world (body and joint velocity)."""
    eng = rig.sim.engine
    qd, jqd = eng.state.body_qd.numpy().copy(), eng.state.joint_qd.numpy().copy()
    child, start = eng.model.joint_child.numpy(), eng.model.joint_qd_start.numpy()
    for b in rig.bodies:
        first = int(start[int(np.flatnonzero(child == b)[0])])
        qd[b][2] = -v0
        if rig.kind == "hinge":                    # v = w x r, r = (-ARM, 0, 0) from the axis: v_z = w_y ARM
            qd[b][4] = jqd[first] = -v0 / ARM
        else:
            jqd[first + (0 if rig.kind == "slider" else 2)] = -v0
    eng.state.body_qd.assign(qd)
    eng.state.joint_qd.assign(jqd)


def mujoco_w(rig: Rig) -> np.ndarray:
    """W of the platen from MuJoCo itself: the translational block of J M^-1 J^T at its centre of mass at qpos0
    (1/kg). W f dt is the velocity MuJoCo gives the body for a force f at its centre of mass."""
    import mujoco  # noqa: PLC0415

    solver = rig.sched._entries["mujoco"].solver
    mj = solver.mj_model
    table = solver.mjc_body_to_newton.numpy()[0]              # MuJoCo body -> model body (world 0)
    mj_body = int(np.flatnonzero(table == rig.body)[0])
    data = mujoco.MjData(mj)
    data.qpos[:] = mj.qpos0
    mujoco.mj_forward(mj, data)
    jacp, x = np.zeros((3, mj.nv)), np.zeros((3, mj.nv))
    mujoco.mj_jacBodyCom(mj, data, jacp, None, mj_body)
    mujoco.mj_solveM(mj, data, x, jacp)
    return jacp @ x.T


@dataclass
class PassRow:
    """One proxy pass of one exchange, world 0's platen (linear parts, world frame, SI)."""

    dt: float
    f_used: np.ndarray             # the feedback force this pass applied to the source and rewound in VBD (N)
    v_before: np.ndarray           # the proxy's velocity in VBD before its step (m/s): the source's stepped
    #                                velocity plus VBD's catch-up of the pose it last left the proxy at
    v_out: np.ndarray              # the proxy's velocity in VBD after its step (m/s)


@dataclass
class ExchangeTap:
    """Records a ``PassRow`` at every proxy harvest of the VBD entry; ``close`` restores the solver."""

    rig: Rig
    rows: list[PassRow] = field(default_factory=list)
    _inner: Callable[..., None] | None = None

    def __post_init__(self) -> None:
        self._inner = self.rig.vbd.coupling_harvest_proxy_wrenches
        self.rig.vbd.coupling_harvest_proxy_wrenches = self._harvest

    def _harvest(self, to_proxy: Any, out_body_f: Any, *, body_qd_before: Any, state: Any, state_out: Any,
                 contacts: Any, dt: float) -> None:
        body, local = self.rig.body, self.rig.local
        used = out_body_f.numpy()[body][:3].astype(np.float64)               # before the harvest overwrites it
        self._inner(to_proxy, out_body_f, body_qd_before=body_qd_before, state=state, state_out=state_out,
                    contacts=contacts, dt=dt)
        self.rows.append(PassRow(float(dt), used, body_qd_before.numpy()[local][:3].astype(np.float64),
                                 state_out.body_qd.numpy()[local][:3].astype(np.float64)))

    def close(self) -> None:
        self.rig.vbd.coupling_harvest_proxy_wrenches = self._inner


@contextmanager
def inject(rig: Rig, forces: Callable[[int], np.ndarray]) -> Iterator[None]:
    """While open, exchange k starts with the feedback force ``forces(k)`` (N, at the centre of mass, no torque) on
    world 0's platen in place of the harvested one."""
    loop = rig.sched._loop
    run, count = loop.run, [0]

    def with_force(state_in: Any, control: Any, contacts: Any, dt: float) -> None:
        held = rig.mapping.coupling_forces.numpy().copy()
        held[rig.body] = 0.0
        held[rig.body][:3] = forces(count[0])
        rig.mapping.coupling_forces.assign(held)
        count[0] += 1
        run(state_in, control, contacts, dt)

    loop.run = with_force
    try:
        yield
    finally:
        del loop.run                                           # back to the class's method


@dataclass
class Rewind:
    """Answers to a known feedback force, per kg. Column a answers a force along axis a: ``source`` = MuJoCo's
    dv / (dt f) (measured W), ``proxy`` = minus the VBD step's dv / (dt f) (the inverse mass VBD used)."""

    w: np.ndarray                  # W from MuJoCo's model (``mujoco_w``)
    source: np.ndarray             # (3, 3) mean over the exchanges that carried each axis
    proxy: np.ndarray              # (3, 3)
    spread: float                  # largest deviation of one exchange from those means (1/kg)
    drift: float                   # largest |proxy dv| over the passes without a force (m/s): 0 if it falls as its body
    dt: float                      # exchange step (s)
    proxy_mass: float              # the scalar installed in VBD (kg)

    @property
    def scale(self) -> float:
        """lambda_max(W): the body's answer along its most mobile axis (1/kg)."""
        return float(np.linalg.eigvalsh(0.5 * (self.w + self.w.T))[-1])


def rewind_response(rig: Rig, accel: float = 40.0, periods: int = 1) -> Rewind:
    """Free flight, ``periods`` control periods: exchange k carries f = accel / lambda_max(W) along axis k % 3 in its
    first pass. The source's and the proxy's answers come from the difference of the exchange's two passes."""
    w = mujoco_w(rig)
    f_mag = accel / float(np.linalg.eigvalsh(0.5 * (w + w.T))[-1])
    tap = ExchangeTap(rig)
    with inject(rig, lambda k: f_mag * np.eye(3)[k % 3]):
        rig.sim.step(periods)
    tap.close()
    rows = tap.rows
    if len(rows) % 2 or len(rows) < 6:
        raise AssertionError(f"expected two passes per exchange (lagged) and three exchanges or more, got "
                             f"{len(rows)} passes")
    src: list[list[np.ndarray]] = [[], [], []]
    prx: list[list[np.ndarray]] = [[], [], []]
    for k in range(len(rows) // 2):
        p0, p1 = rows[2 * k], rows[2 * k + 1]
        impulse = float((p0.f_used - p1.f_used)[k % 3]) * p0.dt
        src[k % 3].append((p0.v_before - p1.v_before) / impulse)
        prx[k % 3].append(-((p0.v_out - p0.v_before) - (p1.v_out - p1.v_before)) / impulse)
    source, proxy = (np.array([np.mean(c, axis=0) for c in cols]).T for cols in (src, prx))
    spread = max(float(np.abs(np.array(c) - np.mean(c, axis=0)).max()) for cols in (src, prx) for c in cols)
    drift = max(float(np.abs(r.v_out - r.v_before).max()) for r in rows[1::2])
    return Rewind(w, source, proxy, spread, drift, rows[0].dt, rig.proxy_mass)


@dataclass
class Contact:
    """The platen at ``v0`` into a pad it already touches (``build(gap=TOUCH, worlds=2)``), against world 1's
    free-fall reference."""

    first_give: float              # exchange 0, pass 0: the proxy's z velocity change inside VBD's step (m/s). No
    #                                feedback exists yet, so it is the start state and VBD's contact solve alone
    kept_give: float               # exchange 0, the pass it keeps (pass 1: with pass 0's feedback and its rewind)
    first_feedback: float          # the feedback pass 0 used, z (N): 0
    pad_gain: float                # z momentum the pad took over the run (N s; negative: pushed down)
    platen_loss: float             # mass x the platen's velocity change against free fall (N s)
    gives: np.ndarray              # the proxy's z velocity change in VBD per exchange, kept passes (m/s)
    ok: bool                       # sim.check()["ok"]


def contact(rig: Rig, v0: float = 0.1, periods: int = 2) -> Contact:
    """Launch the platen and run ``periods`` control periods (two-world rig). From ``TOUCH`` at 0.1 m/s the contact
    is over within 2 periods (5 ms)."""
    eng = rig.sim.engine
    mp = eng.model.particle_mass.numpy().astype(np.float64)
    pw = eng.model.particle_world.numpy()

    def momenta(world: int) -> tuple[float, float]:
        v = eng.state.particle_qd.numpy()[:, 2].astype(np.float64)
        return float(eng.state.body_qd.numpy()[rig.bodies[world]][2]), float((mp[pw == world] * v[pw == world]).sum())

    launch(rig, v0)
    (va, pa), (vb, pb) = momenta(0), momenta(1)
    tap = ExchangeTap(rig)
    rig.sim.step(periods)
    tap.close()
    (vc, pc), (vd, pd) = momenta(0), momenta(1)
    first, kept = tap.rows[0], tap.rows[1]
    gives = np.array([float((r.v_out - r.v_before)[2]) for r in tap.rows[1::2]])
    return Contact(float((first.v_out - first.v_before)[2]), float((kept.v_out - kept.v_before)[2]),
                   float(first.f_used[2]), (pc - pa) - (pd - pb), rig.mass * ((vc - va) - (vd - vb)), gives,
                   bool(rig.sim.check()["ok"]))
