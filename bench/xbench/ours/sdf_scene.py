# SPDX-License-Identifier: Apache-2.0
"""Scenes of the P3-2 mesh-SDF laws A8.push and A9.thread_rundown, and their pack writer (``sdf_pack.py pack``).

``build`` makes the law's model as Alloy's runner does: ``plan_worlds`` + ``build_model``, every condition in one
model, and the twin's ``make_solver``, which sets its indices. For A8, the twin's own peg / hole meshes come without
texture SDFs, so the build also works on CPU. ``write_law`` writes ``packs/laws/<law>/law.json`` and, per
condition, ``cond.json`` + ``meshes/*.obj`` + ``drive.npz`` from that model (``sdf_pack`` docstring), and refuses
a pack that fails its static check (``sdf_geom``).
"""

from __future__ import annotations

import dataclasses
import json
import logging
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[2]), str(HERE)]

import sdf_geom as G  # noqa: E402
from law_pack import law_twin  # noqa: E402

log = logging.getLogger("xbench.ours")
LAWS = PUSH, THREAD = ("A8.push", "A9.thread_rundown")
OUT = HERE / "packs" / "laws"
SCHEMA = "alloy.xbench.ours.sdf/1"
_FRAMES = "a frame every physics step (record_every 1), frame 0 the initial state; W axis = cond.json worlds"
_BODY = {"body_q": "[T, W, B, 7] float32: body origin + quaternion x y z w, world frame (contract)",
         "body_qd": "[T, W, B, 6] float32: COM linear + angular velocity, world frame (contract)"}
EXPORT = {
    PUSH: {"body_names": ["peg"], "frames": _FRAMES, "channels": _BODY,
           "scored_by": "Alloy's GapMonitor on the peg's poses: peg surface samples every 0.5 mm against the hole "
                        "mesh (exact point-mesh distance), max penetration from the ramp start (t = 0.1 s) on"},
    THREAD: {"body_names": ["nut"], "frames": _FRAMES, "channels": {
        **_BODY,
        "joint_q": "[T, W, 2] float64, optional: the nut's axial position z (m) and spin angle theta (rad) about the "
                   "bolt axis +z, continuous (unwrapped; theta grows as the nut runs down). Without it, z and theta "
                   "come from body_q (Newton eval_ik, theta unwrapped over frames)",
        "drive_tau": "[T, W] float64, required: the torque (N m) the spin velocity drive applied to the nut about +z "
                     "over the step that ends at the frame (frame 0: 0), i.e. the engine's applied joint drive "
                     "torque. Alloy: kd (target - omega) with omega after the step (its implicit velocity actuator)"},
        "scored_by": "Alloy's ThreadA9Twin: record (z, theta), the torque channel from drive_tau, scalars "
                     "(thread_metrics: advance per rev and mean drive torque over the run-down after 0.25 s)"}}
ENGINE_COLLISION = ("Load each OBJ as a triangle mesh with the engine's own mesh-SDF collision at its documented "
                    "recommended settings for the scene type; record the SDF resolution / spacing, contact offsets / "
                    "margins and their sources in meta.settings. Do not re-hull, decimate or decompose. A static part "
                    "may use the engine's static triangle-mesh path if its SDF contact needs that (say so). Only the "
                    "listed contact pair collides.")


@dataclass
class Built:
    spec: Any
    twin: Any
    worlds: list
    model: Any


def twin_of(law_id: str, duration_s: float | None = None):
    """(LawSpec, the law's own twin); A8's ``CudaOnly`` wrapper is unwrapped (the run below checks the device)."""
    spec, twin = law_twin(law_id)
    if duration_s is not None:
        spec = dataclasses.replace(spec, duration_s=duration_s)
    return spec, getattr(twin, "inner", twin)


def _plain_meshes(twin) -> None:
    """A8: the twin's own peg / hole meshes without texture SDFs. The pack and GapMonitor read triangles only, so
    this works on CPU."""
    from unittest import mock

    with mock.patch("alloy.contact.law_twins.build_feature_sdf", lambda *a, **k: None):
        twin.meshes()


def build(law_id: str, device: str = "cpu") -> Built:
    """The law's model with every condition, as Alloy's runner builds it, with the twin's solver hooks set."""
    from alloy.bench.runner import RunContext, _call_make_solver, build_model, plan_worlds

    spec, twin = twin_of(law_id)
    if hasattr(twin, "meshes"):
        _plain_meshes(twin)
    ctx = RunContext(spec, device, 0, "mujoco", False)
    worlds = plan_worlds(spec, twin.conditions(False), ctx, getattr(twin, "expand", None))
    model = build_model(spec, twin, worlds, ctx)
    _call_make_solver(twin, model, "mujoco")
    return Built(spec, twin, worlds, model)


def state0(b: Built):
    from alloy._vendor import newton

    s0 = b.model.state()
    newton.eval_fk(b.model, b.model.joint_q, b.model.joint_qd, s0)
    if hasattr(b.twin, "init_state"):
        b.twin.init_state(b.model, s0)
    return s0


# ---------------------------------------------------------------- pack: facts ---------------------------------------

def _f(x) -> list:
    """float32 values as exact float64 lists."""
    return np.asarray(x, np.float64).tolist()


def _name(label) -> str:
    return str(label).rsplit("/", 1)[-1]


def _bodies(model, w: int) -> list[dict]:
    bw, m = model.body_world.numpy(), model.body_mass.numpy()
    com, inertia = model.body_com.numpy(), model.body_inertia.numpy()
    return [{"name": _name(model.body_label[b]), "mass_kg": float(m[b]), "com_m": _f(com[b]),
             "inertia_kg_m2": _f(inertia[b]), "note": "inertia about the COM in the body frame; explicit, never "
             "recomputed from the OBJ"} for b in np.flatnonzero(bw == w)]


def _dof(model, d: int, kind: str) -> dict:
    from alloy._vendor import newton

    a = {k: getattr(model, f"joint_{k}").numpy()[d] for k in ("axis", "armature", "damping", "friction", "limit_lower",
                                                                "limit_upper", "target_mode", "target_ke", "target_kd",
                                                                "effort_limit")}
    return {"kind": kind, "axis": _f(a["axis"]), "armature": float(a["armature"]), "damping": float(a["damping"]),
            "friction": float(a["friction"]), "limit": [float(a["limit_lower"]), float(a["limit_upper"])],
            "drive": {"mode": newton.JointTargetMode(int(a["target_mode"])).name, "ke": float(a["target_ke"]),
                      "kd": float(a["target_kd"]), "effort_limit": float(a["effort_limit"])}}


def _joints(model, w: int) -> list[dict]:
    from alloy._vendor import newton

    bw, child, parent = model.body_world.numpy(), model.joint_child.numpy(), model.joint_parent.numpy()
    qd0, dim, jt = model.joint_qd_start.numpy(), model.joint_dof_dim.numpy(), model.joint_type.numpy()
    xp, xc = model.joint_X_p.numpy(), model.joint_X_c.numpy()
    out = []
    for j in np.flatnonzero(bw[child] == w):
        nl = int(dim[j][0])
        dofs = [] if jt[j] == newton.JointType.FREE else [
            _dof(model, int(d), "linear" if i < nl else "angular") for i, d in enumerate(range(qd0[j], qd0[j + 1]))]
        out.append({"name": _name(model.joint_label[j]), "type": newton.JointType(int(jt[j])).name,
                    "parent": "world" if parent[j] < 0 else _name(model.body_label[parent[j]]),
                    "child": _name(model.body_label[child[j]]), "X_p": _f(xp[j]), "X_c": _f(xc[j]), "dofs": dofs,
                    "note": "X_p / X_c = x y z qx qy qz qw; FREE: 6 undriven DOF; D6: the listed DOF only, all "
                            "others locked"})
    return out


def _initial(model, s0, w: int) -> dict:
    bw = model.body_world.numpy()
    q, qd = s0.body_q.numpy(), s0.body_qd.numpy()
    jq, jqd, qs, ds = s0.joint_q.numpy(), s0.joint_qd.numpy(), model.joint_q_start.numpy(), model.joint_qd_start.numpy()
    child = model.joint_child.numpy()
    return {"bodies": {_name(model.body_label[b]): {"q": _f(q[b]), "qd": _f(qd[b])} for b in np.flatnonzero(bw == w)},
            "joints": {_name(model.joint_label[j]): {"q": _f(jq[qs[j]:qs[j + 1]]), "qd": _f(jqd[ds[j]:ds[j + 1]])}
                       for j in np.flatnonzero(bw[child] == w)}}


def _shape(model, s: int, name: str, obj: str) -> dict:
    body = int(model.shape_body.numpy()[s])
    return {"name": name, "body": "world" if body < 0 else _name(model.body_label[body]), "obj": obj,
            "pos": [0.0, 0.0, 0.0], "quat_xyzw": [0.0, 0.0, 0.0, 1.0], "mu": float(model.shape_material_mu.numpy()[s]),
            "restitution": float(model.shape_material_restitution.numpy()[s]),
            "alloy_shape": _name(model.shape_label[s])}


def _pair(shapes: list[dict]) -> list[dict]:
    a, b = shapes
    return [{"shapes": [a["name"], b["name"]], "mu": max(a["mu"], b["mu"]), "restitution": 0.0,
             "note": "the friction Alloy's MuJoCo rows use (geom mu mixed by max); Coulomb, static = dynamic, "
                     "elliptic cone, no torsional / rolling friction, no restitution (MuJoCo soft contact)"}]


# ---------------------------------------------------------------- pack: geometry, drive, Alloy's collision ----------

def _geometry(law_id: str, b: Built, d: Path, w: int, nut_pose: np.ndarray | None) -> tuple[list[dict], dict]:
    """Write the OBJs of world ``w`` (the condition's first) into ``d/meshes``; returns (shapes, the static check)."""
    if law_id == PUSH:
        meshes = G.shape_meshes(b.model, w)
        shapes = [_shape(b.model, m["shape"], m["name"], f"meshes/{m['name']}.obj") for m in meshes]
        for sh, m in zip(shapes, meshes, strict=True):
            G.write_obj(d / sh["obj"], m["verts"], m["tris"], [f"Alloy A8.push {m['name']}: "
                        "alloy.testing.contact (CON-46), body frame, metres, Alloy's float32 values exactly"], 17)
        return shapes, {"meshes": G.check_meshes(d, shapes, meshes)}
    s0, s1 = b.model.shape_world_start.numpy()[w:w + 2]
    ids = {_name(b.model.shape_label[s]): s for s in range(int(s0), int(s1))}
    shapes = []
    for name, (v, f, header) in G.thread_meshes().items():
        shapes.append(_shape(b.model, ids[name.split("_")[0]], name, f"meshes/{name}.obj"))
        G.write_obj(d / shapes[-1]["obj"], v, f, [f"Alloy A9.thread_rundown {name}: {header}"])
    return shapes, G.check_threads(d, shapes, nut_pose)


def _push_gaps(b: Built, d: Path, shapes: list[dict], rec: dict, gs: list[int]) -> dict:
    """A8: per-world initial peg-vertex gap to the hole, from the pack and from Alloy's model and state."""
    s0, bw = state0(b), b.model.body_world.numpy()
    ours = {s["name"]: G.read_obj(d / s["obj"]) for s in shapes}
    pack = G.posed_gaps(ours["peg"], ours["hole"], [np.asarray(i["bodies"]["peg"]["q"]) for i in rec["initial"]])
    ref = []
    for g in gs:
        m = {x["name"]: (x["verts"], x["tris"]) for x in G.shape_meshes(b.model, g)}
        ref += G.posed_gaps(m["peg"], m["hole"], [s0.body_q.numpy()[np.flatnonzero(bw == g)[0]].astype(np.float64)])
    return {"gap_pack_m": pack, "gap_alloy_m": ref, "max_diff_m": float(np.max(np.abs(np.subtract(pack, ref))))}


def _drive_arrays(law_id: str, b: Built, steps: int) -> dict[str, np.ndarray]:
    """The twin's own drive per step k at t = k dt, every world of the law."""
    st, control = state0(b), b.model.control()
    if law_id == PUSH:
        from law_pack import _drive

        bw = b.model.body_world.numpy()
        body_map = np.array([[np.flatnonzero(bw == w)[0]] for w in range(len(b.worlds))])
        return {"body_f": _drive(b.twin, b.model, st, steps, b.spec.dt, body_map)}
    spin = b.twin._drive_in.qd_spin.numpy()
    jf, tq = np.zeros((steps, len(spin), 2), np.float32), np.zeros((steps, len(spin)), np.float32)
    for k in range(steps):
        b.twin.drive(k, k * b.spec.dt, st, control)
        f = control.joint_f.numpy()
        jf[k], tq[k] = np.stack([f[spin - 1], f[spin]], -1), control.joint_target_qd.numpy()[spin]
    return {"joint_f": jf, "target_qd": tq}


def _drive_doc(law_id: str, b: Built, cond) -> dict:
    if law_id == PUSH:
        p = b.twin.plan
        return {"kind": "body force on the peg COM, world frame", "body": "peg",
                "profile": f"F(t) = (0, 0, -{p.force_n:g} N) * clamp((t - {p.settle_s:g}) / {p.ramp_s:g}, 0, 1): "
                           f"settle {p.settle_s:g} s, ramp {p.ramp_s:g} s, hold {p.hold_s:g} s; t = k dt over step k",
                "file": "drive.npz", "arrays": {"body_f": "[steps, W, 1, 6] float32: force at the COM then torque, "
                                                          "world frame (Newton State.body_f), the twin's own push"}}
    from alloy.contact.thread_twins import KD_SPIN, SPIN_RAMP_S

    return {"kind": "joint drive on nut_drive (a D6: the axial slide and the spin about the bolt axis +z)",
            "axial": f"constant joint force -{cond.params['F']:g} N on the slide from t = 0 (opposes the advance)",
            "spin": f"velocity drive, kd = {KD_SPIN:g} N m s/rad, target omega(t) = {cond.params['omega']:.9g} rad/s "
                    f"* smoothstep(t / {SPIN_RAMP_S:g} s), smoothstep(s) = s^2 (3 - 2 s) on [0, 1]; t = k dt over "
                    "step k; implicit (MuJoCo implicitfast velocity actuator), effort limit as in joints",
            "file": "drive.npz", "arrays": {"joint_f": "[steps, W, 2] float32: feed-forward joint force (slide N, "
                                                       "spin N m) over step k", "target_qd": "[steps, W] float32: "
                                                       "the spin drive's target velocity (rad/s) over step k"}}


def _alloy_collision(law_id: str, b: Built) -> dict:
    rec = {"solver_settings": b.twin.settings_record("mujoco")}
    if law_id == PUSH:
        return {"representation": "Newton texture SDFs baked from both OBJ meshes (CUDA), an 'sdf' precision pair "
                "appended to SolverMuJoCo; see solver_settings.precision_pair (voxel_size, narrow_band, gap, "
                "margin, pair_stiffness, mj_imp)", "checker": "GapMonitor, sample spacing "
                f"{b.twin.monitor_spacing:g} m, exact point-mesh distances (never the SDF)", **rec}
    from alloy.contact import analytic_pair as AP
    from alloy.contact import thread_twins as TT

    pair = TT.thread_pairs(TT.row_stiffness(b.spec.dt))[0]
    return {"representation": "analytic: the bolt's ISO thread SDF (no voxels, no texture) against samples on the "
            "nut's thread flanks; the MuJoCo cylinders 'bolt' / 'nut' are filtered from collision",
            "pair": dataclasses.asdict(pair.spec), "bolt": dataclasses.asdict(pair.partner),
            "nut": dataclasses.asdict(pair.sampled), "row_stiffness_n_per_m": pair.spec.pair_stiffness,
            "nut_samples": f"{TT.SAMPLE_SPACING:g} m grid on the middle {TT.NUT_FLANK_BAND:.0%} of each flank, from "
                           f"{AP.END_TRIM:g} pitch inside each nut face, {TT.TWIN_SECTORS} sectors",
            "pack_meshes": f"thread_sdf.thread_mesh, {G.THREAD_SEGMENTS} segments per turn; bolt cut to "
                           f"{G.BOLT_PACK_Z} m (Alloy's analytic bolt: {TT.BOLT_Z} m)", **rec}


# ---------------------------------------------------------------- pack: writer --------------------------------------

def _record(law_id: str, b: Built, cond, gs: list[int], steps: int) -> dict:
    s0, model, spec = state0(b), b.model, b.spec
    return {"schema": SCHEMA, "law": law_id, "condition": cond.id, "params": dict(cond.params), "dt": spec.dt,
            "substeps": spec.substeps, "duration_s": spec.duration_s, "steps": steps, "record_every": 1,
            "gravity": _f(model.gravity.numpy()[gs[0]]), "worlds": len(gs),
            "samples": [dict(b.worlds[g].sample) for g in gs], "bodies": _bodies(model, gs[0]),
            "joints": _joints(model, gs[0]), "initial": [_initial(model, s0, g) for g in gs],
            "drive": _drive_doc(law_id, b, cond), "alloy_collision": _alloy_collision(law_id, b),
            "engine_collision": ENGINE_COLLISION, "export": EXPORT[law_id]}


def _initial_err(b: Built, rec: dict, gs: list[int]) -> float:
    """Max |initial body q, qd read back from cond.json - Alloy's state| over the condition's worlds."""
    s0, bw = state0(b), b.model.body_world.numpy()
    q, qd = s0.body_q.numpy(), s0.body_qd.numpy()
    err = 0.0
    for init, g in zip(rec["initial"], gs, strict=True):
        for bi in np.flatnonzero(bw == g):
            got = init["bodies"][_name(b.model.body_label[bi])]
            err = max(err, float(np.abs(np.float32(got["q"]) - q[bi]).max()),
                      float(np.abs(np.float32(got["qd"]) - qd[bi]).max()))
    return err


def _failures(law_id: str, chk: dict) -> list[str]:
    bad = [f"initial state off by {chk['initial_err']:.3g}"] if chk["initial_err"] > 0.0 else []
    if law_id == THREAD:
        return bad + G.thread_failures(chk["threads"])
    bad += [f"{k}: OBJ off Alloy's mesh by {v['vertex_err_m']:.3g} m" for k, v in chk["meshes"].items()
            if not (v["triangles_equal"] and v["vertex_err_m"] <= G.OBJ_TOL_M)]
    if chk["gaps"]["max_diff_m"] > G.OBJ_TOL_M:
        bad.append(f"initial peg-hole gaps differ from Alloy's by {chk['gaps']['max_diff_m']:.3g} m")
    return bad


def write_condition(law_id: str, b: Built, cond, gs: list[int], d: Path, steps: int,
                    drive: dict[str, np.ndarray]) -> dict:
    """One condition dir; raises ``ValueError`` if the written pack fails its static check."""
    d.mkdir(parents=True, exist_ok=True)
    rec = _record(law_id, b, cond, gs, steps)
    nut = rec["initial"][0]["bodies"].get("nut")
    rec["shapes"], geo = _geometry(law_id, b, d, gs[0], None if nut is None else np.asarray(nut["q"]))
    rec["contact_pairs"] = _pair(rec["shapes"])
    arrays = {k: v[:, gs] for k, v in drive.items()}
    np.savez_compressed(d / "drive.npz", body_names=np.array(EXPORT[law_id]["body_names"]), **arrays)
    (d / "cond.json").write_text(json.dumps(rec, indent=1, default=str))
    disk = json.loads((d / "cond.json").read_text())
    chk = {"initial_err": _initial_err(b, disk, gs), **({"threads": geo} if law_id == THREAD else geo)}
    if law_id == PUSH:
        chk["gaps"] = _push_gaps(b, d, disk["shapes"], disk, gs)
    rec["checks"] = chk
    (d / "cond.json").write_text(json.dumps(rec, indent=1, default=str))
    bad = _failures(law_id, chk)
    if bad:
        raise ValueError(f"{law_id}/{cond.id}: pack check failed: {bad}")
    log.info("%s/%s: %d worlds, check %s", law_id, cond.id, len(gs), json.dumps(chk, default=str)[:400])
    return {"id": cond.id, "params": dict(cond.params), "worlds": len(gs)}


def write_law(law_id: str, out: Path) -> Path:
    """``out/<law>/law.json`` and every condition dir (module docstring)."""
    b = build(law_id)
    d = out / law_id
    shutil.rmtree(d, ignore_errors=True)
    spec = b.spec
    steps = max(1, int(round(spec.duration_s / spec.dt)))
    drive = _drive_arrays(law_id, b, steps)
    conds = [write_condition(law_id, b, c, [g for g, w in enumerate(b.worlds) if w.condition.id == c.id], d / c.id,
                             steps, drive) for c in b.twin.conditions(False)]
    rec = {"law": law_id, "schema": SCHEMA, "dt": spec.dt, "substeps": spec.substeps, "duration_s": spec.duration_s,
           "steps": steps, "metrics": {k: {"abs": t.abs, "rel": t.rel, "source": t.source}
                                       for k, t in spec.metrics.items()}, "stat": dict(spec.stat),
           "conditions": conds, "export": EXPORT[law_id],
           "score": f"python bench/xbench/ours/sdf_pack.py score RUN_DIR --laws {law_id} --device cuda:0 reads "
                    f"RUN_DIR/{law_id}/<condition>.npz -> RUN_DIR/{law_id}.scored.json"}
    (d / "law.json").write_text(json.dumps(rec, indent=1, default=str))
    log.info("%s: %d conditions -> %s", law_id, len(conds), d)
    return d
