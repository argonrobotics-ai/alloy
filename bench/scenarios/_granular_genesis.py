# SPDX-License-Identifier: Apache-2.0
"""Offline Genesis SPH reference for G02 (never in the simulation loop; runs in its own venv without Newton).

    ~/gvenv/bin/python -m bench.scenarios._granular_genesis --cond dam_break --solver WCSPH --particle 0.0025
    ~/gvenv/bin/python -m bench.scenarios._granular_genesis --cond settle --solver DFSPH --particle 0.005

Genesis World 1.4.5 @983bd0dd SPH (SCENARIOS.md §5): the G02 geometry and measures, one condition per process.
- dam_break: water column a = 57.15 mm wide, 2a high, at the closed end of a 0.6 m channel; front
  Z = 1 + shift of p99.5(x) / a every frame to t = 0.25 s, NRMSE vs the Martin & Moyce table of granular_laws (same
  UNVERIFIED transcription as the Newton rows), Ritter count.
- settle: the same column in a closed box of width a for 5 s: volume loss 1 - h/h0 with h = p99.5(z), bottom-layer
  (z < particle size) mean SPH pressure vs rho g (h_surface - z) with the current surface, density error.
The domain box (Genesis CubeBoundary: centre clamp, zero restitution, no boundary particles) is the channel; the slab is
WIDTH wide in y (not quasi-2D like the MPM slab: Genesis has no boundary particles, so a thin slab would be mostly
wall-deficient particles). Rows append to bench/results/granular/G02_dam_break.ref.jsonl, solver "ref:genesis_sph".
"""
from __future__ import annotations

import argparse
import json
import math
import time
from importlib import metadata
from pathlib import Path

import numpy as np

from bench import harness as H
from bench.scenarios.granular_laws import MM_T, MM_Z, G, nrmse, ritter_front

A, CHANNEL, T_FRONT, WIDTH, RHO = 0.05715, 0.6, 0.25, 0.05, 1000.0


class _G02Ref(H.Scenario):
    id = "G02_dam_break"
    family = "granular"
    title = "G02 offline reference: Genesis SPH"
    accuracy_unit = "nrmse"
    accuracy_ref = ("dam_break: front Z(T) NRMSE vs Martin & Moyce 1952 n^2=2 (table UNVERIFIED); settle: no headline "
                    "(metrics volume_loss_pct, pressure_err_pct)")


def device_sync() -> None:
    import torch

    torch.cuda.synchronize()
    try:
        import quadrants

        quadrants.sync()
    except Exception:  # noqa: BLE001 - best effort, the torch sync already drained torch's stream
        pass


def build(a: argparse.Namespace):
    import genesis as gs

    gs.init(backend=gs.gpu, precision="32", logging_level="warning")
    x_max = CHANNEL if a.cond == "dam_break" else A
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=a.dt, substeps=a.substeps, gravity=(0.0, 0.0, -G)),
        sph_options=gs.options.SPHOptions(particle_size=a.particle, pressure_solver=a.solver,
                                          lower_bound=(0.0, -0.5 * WIDTH, 0.0), upper_bound=(x_max, 0.5 * WIDTH, 0.4)),
        show_viewer=False)
    liquid = scene.add_entity(material=gs.materials.SPH.Liquid(rho=RHO, sampler="regular"),
                              morph=gs.morphs.Box(pos=(0.5 * A, 0.0, A), size=(A, WIDTH, 2.0 * A)))
    scene.build()
    return scene, liquid


def positions(liquid) -> np.ndarray:
    return liquid.get_particles_pos().detach().cpu().numpy().reshape(-1, 3).astype(np.float64)


def reordered(scene, name: str) -> np.ndarray:
    from genesis.utils.misc import qd_to_numpy

    return np.asarray(qd_to_numpy(getattr(scene.sph_solver.particles_reordered, name), transpose=True)[0], np.float64)


def run(a: argparse.Namespace, rec: dict) -> None:
    t0 = time.perf_counter()
    scene, liquid = build(a)
    device_sync()
    rec["setup_s"] = time.perf_counter() - t0
    q0 = positions(liquid)
    x0, z0 = float(np.percentile(q0[:, 0], 99.5)), float(np.percentile(q0[:, 2], 99.5))
    total = int(round(rec["horizon_s"] / a.dt))
    front, level, timed, stable = [], [], 0.0, True
    for k in range(1, total + 1):
        t1 = time.perf_counter()
        scene.step()
        device_sync()
        timed += time.perf_counter() - t1
        t = k * a.dt
        sample_front = a.cond == "dam_break" and t <= T_FRONT + 1e-9
        sample_level = a.cond == "settle" and k % max(1, int(round(0.1 / a.dt))) == 0
        if not (sample_front or sample_level or k == total):
            continue
        q = positions(liquid)
        if not np.isfinite(q).all():
            stable = False
            break
        if sample_front:
            front.append((t, 1.0 + (float(np.percentile(q[:, 0], 99.5)) - x0) / A))
        if sample_level:
            level.append((t, float(np.percentile(q[:, 2], 99.5)) / z0))
    rec.update(steps=k, complete=k >= total and stable, stable=stable, ms_per_step=1e3 * timed / k,
               world_steps_per_s=k / timed, sim_seconds_per_gpu_second=k * a.dt / timed,
               single_world_realtime_factor=k * a.dt / timed)
    n = int(scene.sph_solver.n_particles)
    rec["metrics"].update(particles=n, cond=a.cond, particle_size_m=a.particle, slab_width_m=WIDTH,
                          substep_s=a.dt / a.substeps, pressure_solver=a.solver,
                          escaped_below_floor=int(np.sum(positions(liquid)[:, 2] < -a.particle)))
    if not stable:
        return
    (_dam if a.cond == "dam_break" else _settle)(scene, liquid, a, rec, front, level, z0)


def _dam(scene, liquid, a, rec, front, level, z0) -> None:
    tz = np.array(front)
    T = tz[:, 0] * math.sqrt(2 * G / A)
    use = MM_T <= T.max()
    nr = nrmse(np.interp(MM_T[use], T, tz[:, 1]), MM_Z[use]) if use.sum() >= 3 else None
    viol = int(np.sum(tz[:, 1] > ritter_front(tz[:, 0], A, 2 * A) / A))
    rec["accuracy_error"] = nr
    rec["metrics"].update(front_nrmse=nr, front_points=int(use.sum()), ritter_violations=viol,
                          front_Z_at_T=[(round(float(t), 3), round(float(z), 3)) for t, z in zip(T[::5], tz[::5, 1])],
                          law_checks=dict(front=nr is not None and nr <= 0.1, ritter=viol == 0))


def _settle(scene, liquid, a, rec, front, level, z0) -> None:
    pos, p, rho = reordered(scene, "pos"), reordered(scene, "p"), reordered(scene, "rho")
    h0 = 2.0 * A
    h_now = float(np.percentile(pos[:, 2], 99.5))
    ratio = h_now / z0
    bottom = pos[:, 2] < a.particle
    surface = h_now + (h0 - z0)
    p_bottom = float(np.mean(p[bottom]))
    p_law = RHO * G * (surface - float(np.mean(pos[bottom, 2])))
    mid = (pos[:, 2] > a.particle) & (pos[:, 2] < 0.8 * h_now)
    slope = float(np.polyfit(pos[mid, 2], p[mid], 1)[0]) if mid.sum() > 10 else math.nan
    vol, prs = 100.0 * (1.0 - ratio), 100.0 * (p_bottom - p_law) / p_law
    rec["metrics"].update(h_ratio=ratio, volume_loss_pct=vol, p_bottom_pa=p_bottom, p_law_pa=p_law,
                          pressure_err_pct=prs, pressure_err_vs_h0_pct=100.0 * (p_bottom / (RHO * G * h0) - 1.0),
                          hydro_slope_err_pct=100.0 * (-slope / (RHO * G) - 1.0),
                          density_err_pct_mean=100.0 * (float(np.mean(rho)) / RHO - 1.0),
                          bottom_layer_particles=int(bottom.sum()),
                          level_trace=[(round(t, 2), round(r, 4)) for t, r in level[::5]],
                          law_checks=dict(pressure=abs(prs) <= 3.0, volume=vol <= 1.0))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cond", choices=("dam_break", "settle"), required=True)
    ap.add_argument("--solver", choices=("WCSPH", "DFSPH"), default="WCSPH")
    ap.add_argument("--particle", type=float, default=0.0025, help="SPH particle size [m]")
    ap.add_argument("--dt", type=float, default=2e-3, help="frame dt [s] (measure cadence)")
    ap.add_argument("--substeps", type=int, default=40)
    ap.add_argument("--horizon", type=float, default=None, help="default 0.3 s dam_break, 5 s settle")
    ap.add_argument("--out", default=str(H.REPO / "bench" / "results"))
    a = ap.parse_args()
    horizon = a.horizon or (0.3 if a.cond == "dam_break" else 5.0)
    name = f"ref_genesis_{a.solver.lower()}_p{a.particle * 1e3:g}mm_{a.cond}"
    cfg = H.Config(name, "ref:genesis_sph", dt=a.dt, substeps=a.substeps, cuda_graph=False, max_nworld=1,
                   params=dict(particle_size=a.particle, pressure_solver=a.solver, slab_width=WIDTH, cond=a.cond,
                               material="SPH.Liquid defaults (stiffness 5e4, exponent 7, mu 0.005, gamma 0.01)"))
    opts = H.RunOptions(device="cuda:0", horizon_s=horizon, out_dir=Path(a.out),
                        command="python -m bench.scenarios._granular_genesis " +
                        " ".join(f"--{k} {v}" for k, v in vars(a).items() if v is not None and k != "out"))
    rec = H._base_record(_G02Ref(), cfg, 1, opts)
    rec["versions"].update(genesis=_ver("genesis-world"), torch=_ver("torch"), quadrants=_ver("quadrants"),
                           genesis_commit="983bd0dd51f7eb6db690aa180bb1692c03fdba8c")
    try:
        import torch

        rec["device_name"] = torch.cuda.get_device_name(0)
        run(a, rec)
    except Exception as exc:  # noqa: BLE001 - the record is the error report
        rec.update(status="error", error=f"{type(exc).__name__}: {exc}"[:2000])
    path = Path(a.out) / "granular" / "G02_dam_break.ref.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(rec, default=H._json_default) + "\n")
    m = rec["metrics"]
    print(name, rec["status"], rec["error"], f"ms/step={rec['ms_per_step']}", f"nrmse={m.get('front_nrmse')}",
          f"ritter={m.get('ritter_violations')}", f"vol={m.get('volume_loss_pct')}", f"p={m.get('pressure_err_pct')}")
    return 0


def _ver(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


if __name__ == "__main__":
    raise SystemExit(main())
