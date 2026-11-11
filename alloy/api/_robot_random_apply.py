# SPDX-License-Identifier: Apache-2.0
"""Write a scene's robot and gravity draws (``alloy.api._robot_random``) into the built model, per world.

``install(built, scene, seed)`` runs at build, before the solvers are made: it resolves every row to model indices,
keeps a copy of the model's nominal values, draws each row for each world and writes the result. ``attach(engine)``
adds ``redraw`` to the engine's reset hooks: in every reset, rows with ``when="reset"`` get a new draw in the worlds
that reset (the world's reset count is part of the draw's key), the model is rewritten and the solvers are told
what changed. All of it runs on the host, outside any captured graph.

Every write starts from the nominal copy and applies all the rows that touch an item, so the order of the rows does
not matter and a redraw never compounds: mass = nominal x (the product of its mass_scale draws) + (its mass_add
draws); inertia = nominal x mass / nominal mass; centre of mass = nominal + offsets; a servo's kp and kd = nominal x
their factors. Friction, absolute armature and gravity components are set to the drawn value (the last row wins).

Draws are counter-based (``alloy.scene.randomize.draw``): a function of the seed, the global world id, the row's name
and the reset count only, so a world gets the same values on every device and for every process count.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from alloy.api import _robot_random as R
from alloy.api import _robot_random_start as S
from alloy.api import _robot_random_targets as T
from alloy.api.errors import BuildError

log = logging.getLogger("alloy.api")
INERTIAL = ("body_mass", "body_inertia", "body_inv_mass", "body_inv_inertia", "body_com")
NOMINAL = (*INERTIAL, "shape_material_mu", "joint_armature", "joint_target_ke", "joint_target_kd", "gravity")
NOMINAL_MJ = ("actuator_gainprm", "actuator_biasprm")
GROUPS = {"mass_scale": "inertial", "mass_add": "inertial", "com_x": "inertial", "com_y": "inertial",
          "com_z": "inertial", "friction": "friction", "kp_scale": "gains", "kd_scale": "gains",
          "armature_scale": "armature", "armature": "armature", "x": "gravity", "y": "gravity", "z": "gravity",
          **dict.fromkeys(R.START_FIELDS, "start")}
COM_AXES = {"com_x": 0, "com_y": 1, "com_z": 2}


@dataclass
class Applied:
    """What ``install`` keeps on ``built.robot_random``: the rows with their indices, the nominal model values, the
    current draws ``values[key]`` (worlds,) and each world's reset count."""

    rows: list[dict[str, Any]]
    targets: list[T.Target]
    seed: int
    gids: np.ndarray                                   # global world id of each local world
    nominal: dict[str, np.ndarray]
    values: dict[str, np.ndarray] = field(default_factory=dict)
    counts: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    index: Any = None                                  # the NameIndex of the build (start rows place robots by it)

    def groups(self, when: str | None = None) -> list[str]:
        """The groups (inertial, friction, gains, armature, gravity) with a row, in a fixed order."""
        seen = {GROUPS[r["field"]] for r in self.rows if when is None or r["when"] == when}
        return [g for g in ("inertial", "friction", "gains", "armature", "gravity", "start") if g in seen]


def install(built: Any, scene: Any, seed: int) -> None:
    """Resolve, draw and write the scene's rows into ``built.model`` (nothing when the scene has none).

    Example: ``install(built, scene, seed=0)`` after assembly, before the simulation is wired."""
    rows = [dict(r) for r in getattr(scene, "robot_random", [])]
    if not rows:
        return
    from alloy.api.names import NameIndex  # noqa: PLC0415

    model, worlds = built.model, int(built.world_count)
    idx = NameIndex.from_built(built)
    targets = [T.resolve(built, idx, r) for r in rows]
    mj = getattr(model, "mujoco", None)
    nominal = {n: getattr(model, n).numpy().copy() for n in NOMINAL if getattr(model, n, None) is not None}
    nominal.update({n: getattr(mj, n).numpy().copy() for n in NOMINAL_MJ if getattr(mj, n, None) is not None})
    gids = np.asarray([built.global_world_id(w) for w in range(worlds)], np.int64)
    st = Applied(rows, targets, int(seed), gids, nominal, counts=np.zeros(worlds, np.int64), index=idx)
    built.robot_random = st
    every = np.arange(worlds)
    if "inertial" in st.groups():
        _mass_terms(st, every[:1], low=True)             # refuse a range that can empty a link, whatever is drawn
    _draw(st, every, [r for r in rows if r["field"] not in R.START_FIELDS])
    st.values.update({R.key(r): np.zeros(worlds) for r in rows if r["field"] in R.START_FIELDS})   # until a reset
    for group in st.groups():
        WRITERS[group](model, st, every)
    from alloy.scene.assembler import digest_model  # noqa: PLC0415

    built.model_digest = digest_model(model)             # the digest names the model the worlds start from
    log.info("randomized %d robot / gravity fields over %d worlds (seed %d)", len(rows), worlds, seed)


def attach(engine: Any) -> None:
    """Make ``engine.reset`` redraw the ``when="reset"`` rows (nothing when the scene has none).

    Example: ``attach(engine)`` once the simulation is wired."""
    st = getattr(engine.built, "robot_random", None)
    if st is not None and st.groups("reset"):
        engine.reset_hooks.append(lambda world_mask: redraw(engine, world_mask))


def redraw(engine: Any, world_mask: Any = None) -> None:
    """A new draw of every ``when="reset"`` row for the worlds of ``world_mask`` (all when None), written to the
    model and passed to the solvers. A failure puts the previous draws back before it is raised.

    Example: ``redraw(sim.engine)``."""
    from alloy._vendor import newton  # noqa: PLC0415

    st, model = engine.built.robot_random, engine.built.model
    count = len(st.counts)
    worlds = np.arange(count) if world_mask is None else np.flatnonzero(world_mask.numpy()[:count])
    if not len(worlds):
        return
    groups = st.groups("reset")
    before = ({k: v.copy() for k, v in st.values.items()}, st.counts.copy())
    st.counts[worlds] += 1
    try:
        _draw(st, worlds, [r for r in st.rows if r["when"] == "reset"])
        for group in groups:
            WRITERS[group](model, st, worlds)
    except Exception:
        st.values, st.counts = before
        for group in groups:
            WRITERS[group](model, st, worlds)
        raise
    flags = 0
    for name in (n for group in groups for n in FLAGS[group]):
        flags |= int(getattr(newton.ModelFlags, name))
    if flags:
        engine._notify(flags)
    S.place(engine, worlds)                              # start rows: move the reset worlds' robots (none: no-op)


def values(built: Any) -> dict[str, np.ndarray]:
    """{row name: (worlds,) values each world holds now}. Example: ``values(sim.built)["go2.mass_scale"]``."""
    st = getattr(built, "robot_random", None)
    return {} if st is None else {k: v.copy() for k, v in st.values.items()}


def _draw(st: Applied, worlds: np.ndarray, rows: list[dict[str, Any]]) -> None:
    from alloy.scene.randomize import draw  # noqa: PLC0415

    for r in rows:
        k = R.key(r)
        store = st.values.setdefault(k, np.zeros(len(st.counts)))
        for w in worlds:
            n = int(st.counts[w])
            store[w] = draw(st.seed, int(st.gids[w]), k if n == 0 else f"{k}#reset{n}", r["lo"], r["hi"])


def _rows(st: Applied, group: str) -> list[tuple[dict[str, Any], T.Target]]:
    return [(r, t) for r, t in zip(st.rows, st.targets) if GROUPS[r["field"]] == group]


def _assign(model: Any, name: str, host: np.ndarray, holder: Any = None) -> None:
    getattr(holder if holder is not None else model, name).assign(host)


def _mass_terms(st: Applied, worlds: np.ndarray, low: bool = False) -> tuple[np.ndarray, ...]:
    """(template bodies, their global ids [worlds, bodies], mass, centre-of-mass offset) of the inertial rows, from
    the current draws, or from every row's low end when ``low`` (the lightest a link can get)."""
    rows = _rows(st, "inertial")
    bodies = np.unique(np.concatenate([t.bodies for _, t in rows]))
    col = {int(b): i for i, b in enumerate(bodies)}
    scale = np.ones((len(worlds), len(bodies)))
    add = np.zeros_like(scale)
    com = np.zeros((*scale.shape, 3))
    for r, t in rows:
        cols = [col[int(b)] for b in t.bodies]
        v = np.full((len(worlds), 1), r["lo"]) if low else st.values[R.key(r)][worlds][:, None]
        if r["field"] == "mass_scale":
            scale[:, cols] *= v
        elif r["field"] == "mass_add":
            add[:, cols] += v
        else:
            com[:, cols, COM_AXES[r["field"]]] += v
    gid = bodies[None, :] + worlds[:, None] * rows[0][1].stride["body"]
    nominal = st.nominal["body_mass"][gid]
    if low and (np.any(nominal * scale + add <= 0.0) or np.any((nominal <= 0.0) & (add != 0.0))):
        raise BuildError("a randomized link mass can reach zero or less (the low end of mass_add is below minus the "
                         "link's mass, or mass_add is on a massless link)", "raise the low end of the mass_add "
                         "range, or name a link that has mass", field="lo")
    return bodies, gid, nominal * scale + add, com


def _inertial(model: Any, st: Applied, worlds: np.ndarray) -> None:
    _, gid, mass, com = _mass_terms(st, worlds)
    nom = st.nominal
    heavy = nom["body_mass"][gid] > 0.0
    ratio = np.where(heavy, mass / np.where(heavy, nom["body_mass"][gid], 1.0), 1.0)
    host = {n: getattr(model, n).numpy() for n in INERTIAL}
    host["body_mass"][gid] = mass
    host["body_inv_mass"][gid] = nom["body_inv_mass"][gid] / ratio
    host["body_inertia"][gid] = nom["body_inertia"][gid] * ratio[..., None, None]
    host["body_inv_inertia"][gid] = nom["body_inv_inertia"][gid] / ratio[..., None, None]
    host["body_com"][gid] = nom["body_com"][gid] + com
    for n in INERTIAL:
        _assign(model, n, host[n])


def _friction(model: Any, st: Applied, worlds: np.ndarray) -> None:
    host = model.shape_material_mu.numpy()
    for r, t in _rows(st, "friction"):
        gid = t.shapes[None, :] + worlds[:, None] * t.stride["shape"]
        host[gid] = st.values[R.key(r)][worlds][:, None]
    _assign(model, "shape_material_mu", host)


def _factors(st: Applied, group: str, name: str, worlds: np.ndarray, items: str) -> dict[int, np.ndarray]:
    """{template item: (len(worlds),) product of the ``name`` draws that touch it}."""
    out: dict[int, np.ndarray] = {}
    for r, t in _rows(st, group):
        for i in getattr(t, items):
            out.setdefault(int(i), np.ones(len(worlds)))
            if r["field"] == name:
                out[int(i)] = out[int(i)] * st.values[R.key(r)][worlds]
    return out


def _gains(model: Any, st: Applied, worlds: np.ndarray) -> None:
    nom, mj = st.nominal, getattr(model, "mujoco", None)
    stride = _rows(st, "gains")[0][1].stride
    kp = _factors(st, "gains", "kp_scale", worlds, "servo_rows")
    kd = _factors(st, "gains", "kd_scale", worlds, "servo_rows")
    if kp:
        gain, bias = mj.actuator_gainprm.numpy(), mj.actuator_biasprm.numpy()
        velocity = set().union(*[t.velocity_rows for _, t in _rows(st, "gains")])
        for row in kp:
            g = row + worlds * stride["actuator"]
            if row in velocity:                              # gain kv, bias (0, 0, -kv)
                gain[g, 0] = nom["actuator_gainprm"][g, 0] * kd[row]
            else:                                            # gain kp, bias (0, -kp, -kv)
                gain[g, 0] = nom["actuator_gainprm"][g, 0] * kp[row]
                bias[g, 1] = nom["actuator_biasprm"][g, 1] * kp[row]
            bias[g, 2] = nom["actuator_biasprm"][g, 2] * kd[row]
        _assign(model, "actuator_gainprm", gain, mj)
        _assign(model, "actuator_biasprm", bias, mj)
    dof_kp = _factors(st, "gains", "kp_scale", worlds, "target_dofs")
    dof_kd = _factors(st, "gains", "kd_scale", worlds, "target_dofs")
    if dof_kp:
        ke, kdamp = model.joint_target_ke.numpy(), model.joint_target_kd.numpy()
        for d in dof_kp:
            g = d + worlds * stride["dof"]
            ke[g] = nom["joint_target_ke"][g] * dof_kp[d]
            kdamp[g] = nom["joint_target_kd"][g] * dof_kd[d]
        _assign(model, "joint_target_ke", ke)
        _assign(model, "joint_target_kd", kdamp)


def _armature(model: Any, st: Applied, worlds: np.ndarray) -> None:
    host = model.joint_armature.numpy()
    scale = _factors(st, "armature", "armature_scale", worlds, "dofs")
    stride = _rows(st, "armature")[0][1].stride["dof"]
    for d, s in scale.items():
        g = d + worlds * stride
        host[g] = st.nominal["joint_armature"][g] * s
    for r, t in _rows(st, "armature"):
        if r["field"] == "armature":
            gid = t.dofs[None, :] + worlds[:, None] * stride
            host[gid] = st.values[R.key(r)][worlds][:, None]
    _assign(model, "joint_armature", host)


def _gravity(model: Any, st: Applied, worlds: np.ndarray) -> None:
    host = model.gravity.numpy()
    for r, _ in _rows(st, "gravity"):
        host[worlds, R.GRAVITY_AXES[r["field"]]] = st.values[R.key(r)][worlds]
    _assign(model, "gravity", host)


WRITERS = {"inertial": _inertial, "friction": _friction, "gains": _gains, "armature": _armature, "gravity": _gravity,
           "start": lambda model, st, worlds: None}     # start rows change the state, not the model (S.place)
# Newton's flags per group. Gains need two: ACTUATOR_PROPERTIES refreshes servos that read ctrl (<general>),
# JOINT_DOF_PROPERTIES the ones driven from joint targets (<position>, <velocity>).
FLAGS = {"inertial": ("BODY_INERTIAL_PROPERTIES",), "friction": ("SHAPE_PROPERTIES",),
         "gains": ("ACTUATOR_PROPERTIES", "JOINT_DOF_PROPERTIES"), "armature": ("JOINT_DOF_PROPERTIES",),
         "gravity": ("MODEL_PROPERTIES",), "start": ()}

__all__ = ["Applied", "attach", "install", "redraw", "values"]
