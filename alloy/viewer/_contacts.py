# SPDX-License-Identifier: Apache-2.0
"""Rigid contacts of an Alloy simulation as data: where, which way, how hard, and between which named objects.

``read(sim, worlds)`` returns one :class:`ContactSet` per world; ``contacts(sim, world)`` is the public one-world form
(``alloy.viewer.contacts``), a dict of numpy arrays and name lists. :func:`alloy.viewer._contact_draw.draw` puts them
in a Newton viewer.

**What is read.** The contacts of the rigid and robot solver (MuJoCo), as its last physics step left them: after
``sim.step()`` / ``sim.run()`` they are the contacts of the last physics step of the last control period (collision
runs at the start of each step, so positions are where the bodies were then, up to one physics step behind the
state). Both MuJoCo routes give the same fields:

- the CPU route (MuJoCo C, float64; Alloy's default on a CPU): each world's own ``MjData`` (``contact`` array and
  ``mujoco.mj_contactForce``, a read-only function of the step's solved constraint forces);
- MuJoCo Warp (a GPU, or ``ALLOY_CPU_ROUTE=warp`` on a CPU): the pooled contact arrays of the solver's Warp data
  (``nacon``, ``contact.pos / frame / dist / geom / worldid / dim / friction / adhesion / efc_address``) and the
  constraint forces ``efc.force``, decoded on the host as MuJoCo Warp's ``support.contact_force`` does (pyramidal and
  elliptic cones; the net force, adhesion subtracted).

Reading copies to the host and never writes solver state: a run with reads after every step is the same, bit for
bit, as one without.

**Conventions.** ``normal`` is the unit contact normal, pointing from ``body_a`` toward ``body_b`` (MuJoCo's first and
second geom). ``force`` is the contact force acting **on** ``body_b`` in the world frame, newtons: the normal push plus
the friction parts (torsional and rolling torques are not included); the force on ``body_a`` is its negative.
``depth`` is the penetration in metres, positive when the shapes overlap (MuJoCo's ``-dist``). Every contact MuJoCo
lists is returned, including those it makes up to the contact gap before the shapes touch (robot shapes: 5 mm): they
have a negative depth and, while still apart beyond the solver's margin, zero force. ``pos`` is in the world frame
of that world (viewer offsets are not added; the drawing adds them). Names are those of ``sim.names()``: a free
object by its name, a robot part as ``"<robot>/<body>"``, a static object or a part of a fixture by its object name,
a kinematic fixture by its name; a shape Alloy did not name reads ``"world"``.

**Not read** (slice 1): contacts solved outside MuJoCo: cloth and paper or film sheets with anything (VBD), soft
bodies (VBD), liquid particles (SPH), sand and pastes (MPM), rods stepped on ``rod_vbd`` (clamped, yielding or longer
than 40 segments), and precision-fit pairs (pegs, connectors, bolts: their constraint rows are not MuJoCo contacts).
A scene with any of those returns the rigid and robot contacts of its MuJoCo entries and nothing for the rest, with
no error; a scene with no MuJoCo entry returns empty sets. A rod on the MuJoCo chain route (unclamped, or
``route="chain"``) is a chain of MuJoCo bodies: its contacts are read and named by the chain's bodies. An entry that
steps Newton's own MuJoCo C path itself (``use_mujoco_cpu=True``, not built by ``alloy.build``) is skipped.
"""

from __future__ import annotations

import logging
import weakref
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

log = logging.getLogger("alloy.viewer")

WORLD = "world"             # the name of a shape Alloy did not name (no object owns it)
_PYRAMIDAL = 0              # mujoco.mjtCone.mjCONE_PYRAMIDAL (MuJoCo Warp's ConeType.PYRAMIDAL)
_NAMES: weakref.WeakKeyDictionary[Any, _ShapeNames] = weakref.WeakKeyDictionary()


@dataclass
class ContactSet:
    """One world's rigid contacts (module docstring for the conventions); ``n`` contacts, numpy arrays.

    Example: ``s = read(sim, [0])[0]; s.force[s.body_b_mask("block")].sum(axis=0)`` is the push on the block."""

    pos: np.ndarray             # (n, 3) float64, world frame of that world, m
    normal: np.ndarray          # (n, 3) unit, from body_a toward body_b
    force: np.ndarray           # (n, 3) N, world frame, acting on body_b (normal + friction)
    depth: np.ndarray           # (n,) m, positive = penetrating
    body_a: list[str] = field(default_factory=list)
    body_b: list[str] = field(default_factory=list)
    fixed_b: np.ndarray = field(default_factory=lambda: np.zeros(0, bool))   # (n,) body_b cannot move (static)

    @classmethod
    def empty(cls) -> ContactSet:
        z3 = np.zeros((0, 3))
        return cls(z3, z3.copy(), z3.copy(), np.zeros(0), [], [])

    def __len__(self) -> int:
        return int(self.depth.shape[0])

    def body_b_mask(self, name: str) -> np.ndarray:
        """Boolean mask of the contacts whose ``body_b`` is ``name``."""
        return np.array([b == name for b in self.body_b], dtype=bool)

    def as_dict(self) -> dict[str, Any]:
        """``{"pos", "normal", "force", "depth", "body_a", "body_b"}`` (what ``contacts`` returns)."""
        return {"pos": self.pos, "normal": self.normal, "force": self.force, "depth": self.depth,
                "body_a": list(self.body_a), "body_b": list(self.body_b)}

    @classmethod
    def concat(cls, parts: Sequence[ContactSet]) -> ContactSet:
        parts = [p for p in parts if len(p)]
        if not parts:
            return cls.empty()
        if len(parts) == 1:
            return parts[0]
        keys = ("pos", "normal", "force", "depth", "fixed_b")
        cat = {k: np.concatenate([getattr(p, k) for p in parts]) for k in keys}
        return cls(**cat, body_a=[n for p in parts for n in p.body_a], body_b=[n for p in parts for n in p.body_b])


@dataclass
class _Raw:
    """One world's contacts as MuJoCo keeps them: geom pairs, pos, contact frame rows, dist, frame-local force."""

    geom: np.ndarray            # (n, 2) int, the world's MuJoCo geom ids
    pos: np.ndarray             # (n, 3)
    frame: np.ndarray           # (n, 3, 3) rows: normal, tangent 1, tangent 2
    dist: np.ndarray            # (n,)
    force: np.ndarray           # (n, 3) contact-frame force on geom 2 (normal, tangent 1, tangent 2)


# ---- public ---------------------------------------------------------------------------------------------------------
def read(sim: Any, worlds: Sequence[int]) -> dict[int, ContactSet]:
    """The rigid contacts of each world in ``worlds`` (module docstring). Read-only. Example:
    ``read(sim, [0, 1])[1].force``."""
    _check_engine(sim)
    for w in worlds:
        _check_world(sim, w)
    worlds = [int(w) for w in worlds]
    out: dict[int, list[ContactSet]] = {w: [] for w in worlds}
    names = _shape_names(sim)
    for solver in _rigid_solvers(sim.engine):
        g2s = np.asarray(solver.mjc_geom_to_newton_shape.numpy()).reshape(int(sim.engine.model.world_count), -1)
        for w, raw in _read_solver(solver, worlds).items():
            out[w].append(_assemble(raw, g2s[w], names))
    return {w: ContactSet.concat(parts) for w, parts in out.items()}


def contacts(sim: Any, world: int = 0) -> dict[str, Any]:
    """One world's rigid contacts as ``{"pos", "normal", "force", "depth", "body_a", "body_b"}`` (module docstring
    for what is covered and the conventions). Example: ``alloy.viewer.contacts(sim, world=0)["force"]``."""
    return read(sim, [world])[int(world)].as_dict()


def force_on(sim: Any, name: str, world: int = 0) -> np.ndarray:
    """The net rigid-contact force on the object ``name`` in one world, (3,) newtons, world frame: the sum over its
    contacts, each taken with the sign for that object (``force`` where it is ``body_b``, ``-force`` where it is
    ``body_a``). A block resting on a table reads ``(0, 0, m g)``; the table reads the weight it carries, downward.
    A robot or a rod's name counts all its parts (``"gripper"``: every ``"gripper/<link>"``; contacts between two
    of its own parts cancel); a part's name (``"gripper/finger_left"``) counts that part alone.
    Example: ``alloy.viewer.force_on(sim, "block")[2]``."""
    known = sim.names()
    names = set(known.get("bodies", ())) | set(known.get("static", ()))
    if name not in names:
        from alloy.api.errors import UnknownNameError  # noqa: PLC0415 - keeps this module's import light

        raise UnknownNameError(f"no object named {name!r} in this sim", f"pick one of {sorted(names)}", field="name")
    _warn_other_loads(sim)
    found = read(sim, [world])[int(world)]
    on_b = np.array([_belongs(b, name) for b in found.body_b], dtype=bool)
    on_a = np.array([_belongs(a, name) for a in found.body_a], dtype=bool)
    return found.force[on_b].sum(axis=0) - found.force[on_a].sum(axis=0)


def _belongs(body: str, name: str) -> bool:
    """True when a contact's body name is the object ``name`` or one of its parts (``"<name>/<part>"``)."""
    return body == name or body.startswith(name + "/")


#: routes whose loads on a rigid body are exchanged outside MuJoCo's contacts, so force_on does not see them
OTHER_LOADS = ("cloth", "film", "soft", "sph", "liquid", "mpm", "rod_vbd")


def _warn_other_loads(sim: Any) -> None:
    """Once per sim, and only when it has them: say that cloth, soft-body, liquid and sand loads are not counted."""
    if getattr(sim, "_alloy_force_on_noted", False):
        return
    sim._alloy_force_on_noted = True
    routes = sorted({str(r.get("route", "")) for r in sim.routes().values()} if hasattr(sim, "routes") else ())
    other = [r for r in routes if any(k in r for k in OTHER_LOADS)]
    if other:
        log.warning("force_on counts rigid contacts only (MuJoCo's): loads on an object from %s (cloth, soft bodies, "
                    "liquid, sand) are not in it", ", ".join(other))


def draw(backend: Any, sets: dict[int, ContactSet], *, points: bool = True, forces: bool = True) -> None:
    """Draw ``sets`` on a Newton viewer between ``begin_frame`` and ``end_frame`` (:mod:`alloy.viewer._contact_draw`)."""
    from alloy.viewer._contact_draw import draw as draw_sets  # noqa: PLC0415 - it imports ContactSet from here

    draw_sets(backend, sets, points=points, forces=forces)


def _check_engine(sim: Any) -> None:
    if getattr(sim, "engine", None) is not None and getattr(sim, "built", None) is not None:
        return
    from alloy.api.errors import FeatureUnavailableError  # noqa: PLC0415 - keeps this module's import light

    raise FeatureUnavailableError(f"contacts are read from a sim stepped in this process; got {type(sim).__name__}",
                                  "build it with alloy.build(scene) (no processes=) to read its contacts")


def _check_world(sim: Any, world: Any) -> None:
    from alloy.api.errors import SimulationError  # noqa: PLC0415 - keeps this module's import light

    if isinstance(world, bool) or not isinstance(world, (int, np.integer)):
        raise SimulationError(f"world must be a world index (an int), got {world!r}", "pass world=0 for the first")
    if not 0 <= int(world) < int(sim.worlds):
        raise SimulationError(f"world {world} is outside 0..{sim.worlds - 1}", "pass world=0 for the first")


# ---- solvers --------------------------------------------------------------------------------------------------------
def _rigid_solvers(engine: Any) -> list[Any]:
    """The engine's MuJoCo entries (the CPU route's MjcCoreSolver and MuJoCo Warp's SolverMuJoCo)."""
    from alloy._vendor import newton  # noqa: PLC0415

    found = [s for s in getattr(engine, "solvers", ()) if isinstance(s, newton.solvers.SolverMuJoCo)]
    keep = [s for s in found if _is_c_route(s) or not getattr(s, "use_mujoco_cpu", False)]
    if len(keep) < len(found):
        log.debug("contacts: %d MuJoCo entries on Newton's own C step are not read", len(found) - len(keep))
    return keep


def _is_c_route(solver: Any) -> bool:
    return bool(getattr(solver, "use_mujoco_cpu", False)) and hasattr(solver, "datas") and hasattr(solver, "worlds")


def _read_solver(solver: Any, worlds: list[int]) -> dict[int, _Raw]:
    if _is_c_route(solver):
        return {w: _read_c_world(solver, w) for w in worlds if w < len(solver.datas)}
    return _read_warp(solver, worlds)


def _read_c_world(solver: Any, w: int) -> _Raw:
    """One world of the CPU route: its own MjData after the last step; ``mj_contactForce`` only reads."""
    import mujoco  # noqa: PLC0415

    m, d = solver.worlds[w], solver.datas[w]
    n = int(d.ncon)
    c = d.contact
    f6 = np.zeros((n, 6))
    for i in range(n):
        mujoco.mj_contactForce(m, d, i, f6[i])
    return _Raw(np.array(c.geom[:n], np.int64).reshape(n, 2), np.array(c.pos[:n], np.float64).reshape(n, 3),
                np.array(c.frame[:n], np.float64).reshape(n, 3, 3), np.array(c.dist[:n], np.float64).reshape(n),
                f6[:, :3].copy())


def _host(arr: Any, n: int) -> np.ndarray:
    """A host copy of the first ``n`` rows (a CPU Warp array's ``numpy()`` is a live view)."""
    return np.array(arr.numpy()[:n], copy=True)


def _read_warp(solver: Any, worlds: list[int]) -> dict[int, _Raw]:
    """MuJoCo Warp: the pooled contacts, split by world; forces decoded from the constraint rows."""
    d = solver.mjw_data
    n = min(int(d.nacon.numpy()[0]), int(d.naconmax))
    c = d.contact
    world = _host(c.worldid, n)
    geom, pos, dist = _host(c.geom, n), _host(c.pos, n).astype(np.float64), _host(c.dist, n).astype(np.float64)
    frame = _host(c.frame, n).astype(np.float64).reshape(n, 3, 3)
    wanted = np.flatnonzero(np.isin(world, np.asarray(worlds, dtype=world.dtype)))       # decode only these rows
    force = np.zeros((n, 3))
    force[wanted] = _warp_forces(solver, wanted, world[wanted])
    out = {}
    for w in worlds:
        rows = np.flatnonzero(world == w)
        out[w] = _Raw(geom[rows].astype(np.int64), pos[rows], frame[rows], dist[rows], force[rows])
    return out


def _warp_forces(solver: Any, rows: np.ndarray, world: np.ndarray) -> np.ndarray:
    """(len(rows), 3) contact-frame forces of the contacts ``rows`` (their worlds in ``world``): MuJoCo Warp's
    ``contact_force_fn`` (no world-frame map) in numpy."""
    d = solver.mjw_data
    n = rows.size
    if n == 0:
        return np.zeros((0, 3))
    c = d.contact
    efc = np.array(d.efc.force.numpy(), np.float64, copy=True)
    top = int(rows.max()) + 1
    adr, dim = _host(c.efc_address, top)[rows], _host(c.dim, top)[rows]
    mu = _host(c.friction, top)[rows].astype(np.float64)
    adhesion = _host(c.adhesion, top)[rows].astype(np.float64)
    cone = int(np.asarray(solver.mjw_model.opt.cone).ravel()[0])
    njmax = efc.shape[1]
    out = np.zeros((n, 3))
    for i in range(n):
        if adr[i, 0] < 0:
            continue
        row = efc[world[i]]
        f = _pyramid(row, int(adr[i, 0]), mu[i], int(dim[i]), njmax) if cone == _PYRAMIDAL else \
            _elliptic(row, adr[i], int(dim[i]), njmax)
        f[0] -= adhesion[i]
        out[i] = f
    return out


def _pyramid(row: np.ndarray, a: int, mu: np.ndarray, dim: int, njmax: int) -> np.ndarray:
    """Pyramidal cone rows -> (normal, tangent 1, tangent 2), as MuJoCo Warp's ``_decode_pyramid``."""
    f = np.zeros(3)
    if dim == 1:
        f[0] = row[a] if a < njmax else 0.0
        return f
    for k in range(dim - 1):
        p = row[a + 2 * k] if a + 2 * k < njmax else 0.0
        q = row[a + 2 * k + 1] if a + 2 * k + 1 < njmax else 0.0
        f[0] += p + q
        if k < 2:
            f[k + 1] = (p - q) * mu[k]
    return f


def _elliptic(row: np.ndarray, adr: np.ndarray, dim: int, njmax: int) -> np.ndarray:
    """Elliptic cone: one row per component (normal, then the tangents)."""
    f = np.zeros(3)
    for k in range(min(dim, 3)):
        if 0 <= adr[k] < njmax:
            f[k] = row[adr[k]]
    return f


# ---- names and assembly ---------------------------------------------------------------------------------------------
def _assemble(raw: _Raw, geom_to_shape: np.ndarray, names: _ShapeNames) -> ContactSet:
    n = raw.dist.shape[0]
    if n == 0:
        return ContactSet.empty()
    normal = raw.frame[:, 0, :].copy()
    force = np.einsum("ni,nij->nj", raw.force, raw.frame)          # sum_k f_k * frame row k: world frame, on geom 2
    shapes = geom_to_shape[raw.geom]
    second = shapes[:, 1]
    known = (second >= 0) & (second < names.shape_body.size)
    fixed_b = ~known | (names.shape_body[np.where(known, second, 0)] < 0)
    return ContactSet(raw.pos.copy(), normal, force, -raw.dist, [names.of(int(s)) for s in shapes[:, 0]],
                      [names.of(int(s)) for s in second], fixed_b)


@dataclass
class _ShapeNames:
    """Newton shape index -> Alloy name (module docstring), from the sim's name index and object handles."""

    shape_body: np.ndarray
    shape_world: np.ndarray
    body_world: np.ndarray
    stride: tuple[int, int]                         # shapes, bodies per world
    body_name: dict[int, str]                       # template body -> name
    static_name: dict[int, str]                     # template shape -> object name (shapes on no body)
    labels: list[str]

    def of(self, shape: int) -> str:
        if shape < 0 or shape >= self.shape_body.size:
            return WORLD
        body = int(self.shape_body[shape])
        if body >= 0:
            return self.body_name.get(_template(body, self.body_world, self.stride[1]), WORLD)
        name = self.static_name.get(_template(shape, self.shape_world, self.stride[0]))
        if name is not None:
            return name
        head = self.labels[shape].split("/", 1)[0] if shape < len(self.labels) else ""
        return head if head in self.static_name.values() else WORLD


def _template(i: int, world_of: np.ndarray, stride: int) -> int:
    w = int(world_of[i]) if i < world_of.size else -1
    return i - w * stride if w > 0 else i


def _shape_names(sim: Any) -> _ShapeNames:
    engine = sim.engine
    try:
        cached = _NAMES.get(engine)
    except TypeError:
        cached = None
    if cached is not None:
        return cached
    names = _make_names(sim)
    try:
        _NAMES[engine] = names
    except TypeError:
        pass
    return names


def _make_names(sim: Any) -> _ShapeNames:
    m, idx = sim.engine.model, sim.index
    handles = sim.built.handles
    any_h = next(iter(handles.values()), None)
    stride = (len(any_h.world_stride.shape), len(any_h.world_stride.body)) if any_h is not None else (0, 0)
    static: dict[int, str] = {}
    for name, h in handles.items():
        for s in h.template.shape:
            static.setdefault(int(s), name)
    return _ShapeNames(_np(m.shape_body, m.shape_count), _np(m.shape_world, m.shape_count),
                       _np(m.body_world, m.body_count), stride, _body_names(idx, m), static,
                       [str(x) for x in (m.shape_label or [])])


def _np(arr: Any, count: int) -> np.ndarray:
    return np.array(arr.numpy(), copy=True) if arr is not None and count else np.zeros(0, np.int32)


def _body_names(idx: Any, model: Any) -> dict[int, str]:
    """Template body -> the most specific name: a robot part ("arm/link1") over its robot ("arm"), else the shortest
    (a kinematic fixture's own name over its floor part's); bodies the index does not name keep their label."""
    by_body: dict[int, list[str]] = {}
    for name, b in idx.bodies.items():
        by_body.setdefault(int(b), []).append(name)
    out = {b: min(ns, key=lambda s: ("/" not in s, len(s))) for b, ns in by_body.items()}
    labels = [str(x) for x in (model.body_label or [])]
    per_world = len(labels) // max(int(model.world_count), 1)
    for b in range(per_world):
        out.setdefault(b, labels[b].rsplit("/", 1)[-1] if labels[b] else WORLD)
    return out


__all__ = ["WORLD", "ContactSet", "contacts", "draw", "force_on", "read"]
