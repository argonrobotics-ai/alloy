# SPDX-License-Identifier: Apache-2.0
"""P3-2 law port packs: Alloy's rigid analytic laws, one homogeneous scene per condition (Alloy venv).

    PYTHONPATH=. python bench/xbench/ours/law_pack.py [--laws A1.freefall,...] [--out bench/xbench/ours/packs/laws]

Each condition is built exactly as Alloy's runner builds it (``plan_worlds`` + ``build_model`` with the law's own
twin: one world per condition, or the twin's ``expand`` samples, e.g. A12's load factors, which share the geometry
and differ in load), then written as ``packs/laws/<law>/<condition>/``:

- ``scene.xml``: Newton's MJCF of that world (``SolverMuJoCo(save_to_mjcf=...)``): bodies, explicit inertials,
  joints, primitive geoms, the ground plane;
- ``cond.json``: the same compiled facts as the T12 pack (``scene_pack`` helpers) plus what MJCF does not carry:
  per-geom friction and restitution as Newton holds them, the world's gravity, the initial state from the twin's
  ``init_state`` (Newton ``body_q`` x y z qx qy qz qw and ``body_qd`` COM linear + angular, world frame; joint
  coordinates per MuJoCo joint name), dt / steps, and the law's extras (A6: the restitution the law asks for);
- ``drive.npz`` when the twin applies loads: ``body_f`` [steps, W, B, 6] (force at the COM, then torque, world
  frame; Newton ``State.body_f``), the wrench on world w over step k (time ``k * dt``), bodies in ``body_names``
  order. ``initial`` is a list over the condition's W worlds (``worlds``, ``samples``).

``law.json`` beside the conditions holds the LawSpec (dt, duration, metrics, tolerances). D1.rerun reuses the
A3.contact_energy conditions (the engine runs them twice in one process).
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[2]), str(HERE)]

import scene_pack as TP  # noqa: E402

log = logging.getLogger("xbench.ours")
OUT = HERE / "packs" / "laws"
RIGID = ("A1.freefall", "A2.pendulum", "A3.pendulum_energy", "A3.contact_energy", "A4.momentum", "A5.onset",
         "A5.slide", "A5.isotropy", "A6.restitution")
RODS = ("A10.capstan", "A11.hanging_chain", "A11.catenary", "A11.cantilever", "A12.buckling")   # MuJoCo CHAIN route


def law_twin(law_id: str):
    """(LawSpec, fresh twin) from Alloy's registry."""
    from alloy.bench import law as L

    L.discover()
    if law_id not in L._REGISTRY:       # unregistered after its module's one-time import (a test's register/unregister)
        import importlib

        for e in L.CATALOGUE:
            if e.prefix == L.law_prefix(law_id) and e.module in sys.modules:
                importlib.reload(sys.modules[e.module])
    spec, factory = L._REGISTRY[law_id]
    return spec, factory()


def condition_run(law_id: str, cond_index: int, device: str = "cpu"):
    """Model, solver, twin, initial state and worlds of one condition, built as Alloy's runner does."""
    from alloy._vendor import newton

    from alloy.bench.runner import RunContext, _call_make_solver, build_model, plan_worlds

    spec, twin = law_twin(law_id)
    cond = twin.conditions(False)[cond_index]
    ctx = RunContext(spec, device, 0, "mujoco", False)
    worlds = plan_worlds(spec, [cond], ctx, getattr(twin, "expand", None))
    model = build_model(spec, twin, worlds, ctx)
    solver = _call_make_solver(twin, model, "mujoco")
    s0 = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
    if hasattr(twin, "init_state"):
        twin.init_state(model, s0)
    return spec, twin, cond, model, solver, s0, worlds


def _mj_names(m: mujoco.MjModel, kind: mujoco.mjtObj, n: int) -> list[str]:
    return [mujoco.mj_id2name(m, kind, i) for i in range(n)]


def _materials(model, solver, m: mujoco.MjModel) -> dict[str, dict]:
    """Newton friction and restitution per MuJoCo geom name (world 0)."""
    geom_shape = solver.mjc_geom_to_newton_shape.numpy()[0]
    mu, rest = model.shape_material_mu.numpy(), model.shape_material_restitution.numpy()
    names = _mj_names(m, mujoco.mjtObj.mjOBJ_GEOM, m.ngeom)
    return {names[g]: {"mu": float(mu[s]), "restitution": float(rest[s])} for g, s in enumerate(geom_shape) if s >= 0}


def _initial(model, solver, m: mujoco.MjModel, s0, w: int = 0) -> dict:
    """Newton initial state per MuJoCo body / joint name of world ``w``."""
    body_map = solver.mjc_body_to_newton.numpy()[w]
    q, qd = s0.body_q.numpy(), s0.body_qd.numpy()
    bnames = _mj_names(m, mujoco.mjtObj.mjOBJ_BODY, m.nbody)
    bodies = {bnames[b]: {"q": q[nb].tolist(), "qd": qd[nb].tolist()} for b, nb in enumerate(body_map) if nb >= 0}
    jmap = solver.mjc_jnt_to_newton_jnt.numpy()[w]
    jq, jqd = s0.joint_q.numpy(), s0.joint_qd.numpy()
    qs, ds = model.joint_q_start.numpy(), model.joint_qd_start.numpy()
    jnames = _mj_names(m, mujoco.mjtObj.mjOBJ_JOINT, m.njnt)
    joints = {jnames[j]: {"q": jq[qs[nj]:qs[nj + 1]].tolist(), "qd": jqd[ds[nj]:ds[nj + 1]].tolist()}
              for j, nj in enumerate(jmap) if nj >= 0}
    return {"bodies": bodies, "joints": joints, "note": "Newton coordinates: body q = x y z qx qy qz qw; free "
            "joint q = x y z qx qy qz qw; qd = COM linear then angular velocity, world frame"}


def _drive(twin, model, s0, steps: int, dt: float, body_map: np.ndarray) -> np.ndarray | None:
    """Per-step body wrench the twin applies (time-only drives), [steps, W, B, 6] in MuJoCo body order (``body_map``
    [W, B]: Newton body per world and MuJoCo body); None if none."""
    from alloy.bench.analytic.common import BaseTwin

    if type(twin).drive is BaseTwin.drive:
        return None
    st, control = model.state(), model.control()
    st.assign(s0)
    out = np.zeros((steps, *body_map.shape, 6), dtype=np.float32)
    for k in range(steps):
        st.clear_forces()
        twin.drive(k, k * dt, st, control)
        f = st.body_f.numpy()
        out[k] = np.where((body_map >= 0)[..., None], f[np.maximum(body_map, 0)], 0.0)
    return out if np.any(out) else None


def _extras(law_id: str, cond) -> dict:
    if law_id == "A6.restitution":
        return {"restitution_target": float(cond.params["e"]), "note": "Alloy reaches e through its pair-table "
                "damping path (SPEC §5.3 A6); an engine sets its own restitution coefficient to this e"}
    return {}


def write_condition(law_id: str, idx: int, out: Path) -> dict:
    from alloy._vendor import newton

    spec, twin, cond, model, solver, s0, worlds = condition_run(law_id, idx)
    d = out / cond.id
    d.mkdir(parents=True, exist_ok=True)
    newton.solvers.SolverMuJoCo(model, save_to_mjcf=str(d / "scene.xml"))
    m = mujoco.MjModel.from_xml_path(str(d / "scene.xml"))
    md = mujoco.MjData(m)
    mujoco.mj_forward(m, md)
    steps = max(1, int(round(spec.duration_s / spec.dt)))
    body_map = solver.mjc_body_to_newton.numpy()[: len(worlds)]
    drive = _drive(twin, model, s0, steps, spec.dt, body_map)
    if drive is not None:
        np.savez_compressed(d / "drive.npz", body_f=drive,
                            body_names=np.array(_mj_names(m, mujoco.mjtObj.mjOBJ_BODY, m.nbody)))
    rec = {"schema": "alloy.xbench.ours.law/1", "law": law_id, "condition": cond.id, "params": dict(cond.params),
           "dt": spec.dt, "substeps": spec.substeps, "duration_s": spec.duration_s, "steps": steps, "record_every": 1,
           "gravity": model.gravity.numpy()[0].tolist(), "bodies": TP._bodies(m, md), "joints": TP._joints(m),
           "geoms": TP._geoms(m, md, {}), "contact_body_pairs": TP._body_pairs(TP._contact_pairs(m)),
           "worlds": len(worlds), "samples": [dict(w.sample) for w in worlds],
           "materials": _materials(model, solver, m),
           "initial": [_initial(model, solver, m, s0, w) for w in range(len(worlds))],
           "drive": "drive.npz" if drive is not None else None, "extras": _extras(law_id, cond),
           "alloy_settings": twin.settings_record("mujoco") if hasattr(twin, "settings_record") else {}}
    (d / "cond.json").write_text(json.dumps(rec, indent=1, default=str))
    return {"id": cond.id, "params": dict(cond.params), "worlds": len(worlds), "drive": drive is not None}


def write_law(law_id: str, out: Path) -> Path:
    spec, twin = law_twin(law_id)
    d = out / law_id
    shutil.rmtree(d, ignore_errors=True)
    conds = [write_condition(law_id, i, d) for i in range(len(twin.conditions(False)))]
    rec = {"law": law_id, "dt": spec.dt, "substeps": spec.substeps, "duration_s": spec.duration_s,
           "metrics": {k: {"abs": t.abs, "rel": t.rel, "source": t.source} for k, t in spec.metrics.items()},
           "stat": dict(spec.stat), "conditions": conds}
    (d / "law.json").write_text(json.dumps(rec, indent=1, default=str))
    log.info("%s: %d conditions -> %s", law_id, len(conds), d)
    return d


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--laws", default=",".join(RIGID + RODS))
    p.add_argument("--out", type=Path, default=OUT)
    a = p.parse_args()
    for law_id in a.laws.split(","):
        write_law(law_id, a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
