# SPDX-License-Identifier: Apache-2.0
"""Live frames of picked worlds into Rerun while a run steps (SCR-WP13-EXPORT). A tiny logger: the export itself is
plain numpy (``alloy.export``), so any viewer works the same way.

    pip install rerun-sdk
    python examples/export/rerun_live.py                 # opens the Rerun viewer
    python examples/export/rerun_live.py --save run.rrd  # headless: write a recording, open it with `rerun run.rrd`

The training loop only calls ``exporter.capture()`` between steps; a publisher thread polls and logs. ``poll()`` never
blocks, and a frame the publisher has not taken yet is dropped for the newer one, so the run never waits on the view.
"""

from __future__ import annotations

import argparse
import threading

import numpy as np

import alloy
from alloy.export import Frame, FrameExporter


def scene(worlds: int) -> alloy.Scene:
    s = alloy.Scene(worlds=worlds, dt=0.01)
    s.add_ground()
    s.add_box("table", size=(0.6, 0.6, 0.4), pos=(0, 0, 0.2), static=True)
    xs = np.stack([np.linspace(-0.1, 0.1, worlds), np.zeros(worlds), np.full(worlds, 0.7)], axis=1)
    s.add_box("block", size=(0.06, 0.06, 0.06), pos=xs, mass=0.2)
    s.add_cloth("sheet", size=(0.4, 0.4), pos=(0.0, 0.0, 0.9), resolution=12, mass=0.05)
    return s


def log_topology(rr, f: Frame) -> None:
    """Shapes and faces: logged once per world, and again when its topology changes."""
    root = f"world{f.world}"
    for st in f.topology.statics:
        if st["kind"] == "box":
            box = rr.Boxes3D(centers=[st["pos"]], sizes=[st["size"]], quaternions=[rr.Quaternion(xyzw=st["quat_xyzw"])])
            rr.log(f"{root}/static/{st['name']}", box, static=True)
    for name, shapes in zip(f.topology.body_names, f.topology.body_shapes, strict=True):
        for k, sh in enumerate(shapes):
            path = f"{root}/bodies/{name}/shape{k}"
            if sh["kind"] == "box":
                rr.log(path, rr.Boxes3D(centers=[sh["pos"]], sizes=[sh["size"]]), static=True)
            elif sh["kind"] == "sphere":
                rr.log(path, rr.Points3D([sh["pos"]], radii=[sh["radius"]]), static=True)


def log_frame(rr, f: Frame, faces: dict[str, np.ndarray]) -> None:
    root = f"world{f.world}"
    _set_time(rr, f.t)
    for i, name in enumerate(f.topology.body_names if f.topology else _names[f.world]):
        rr.log(f"{root}/bodies/{name}", rr.Transform3D(translation=f.body_pos[i],
                                                       rotation=rr.Quaternion(xyzw=f.body_quat_xyzw[i])))
    for name, verts in f.meshes.items():
        rr.log(f"{root}/meshes/{name}", rr.Mesh3D(vertex_positions=verts, triangle_indices=faces[name]))
    for name, p in f.particles.items():
        rr.log(f"{root}/particles/{name}", rr.Points3D(p["pos"][p["active"]], radii=0.002))
    for name, line in f.rods.items():
        rr.log(f"{root}/rods/{name}", rr.LineStrips3D([line]))


_names: dict[int, list[str]] = {}


def _set_time(rr, t: float) -> None:
    if hasattr(rr, "set_time"):
        rr.set_time("sim", duration=t)
    else:
        rr.set_time_seconds("sim", t)


def publisher(rr, exporter: FrameExporter, stop: threading.Event) -> None:
    """Poll until stopped (then once more for the last frame) and log every frame taken."""
    faces: dict[int, dict[str, np.ndarray]] = {}
    while True:
        last = stop.is_set()
        frames = exporter.poll()
        for f in frames or []:
            if f.topology is not None:
                _names[f.world], faces[f.world] = f.topology.body_names, f.topology.mesh_faces
                log_topology(rr, f)
            log_frame(rr, f, faces[f.world])
        if last:
            return
        if frames is None:
            stop.wait(0.005)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", help="write a .rrd recording instead of opening the viewer")
    ap.add_argument("--seconds", type=float, default=2.0)
    ap.add_argument("--worlds", type=int, default=4)
    args = ap.parse_args()
    import rerun as rr  # noqa: PLC0415

    rr.init("alloy_frame_export", spawn=args.save is None)
    if args.save:
        rr.save(args.save)
    sim = alloy.build(scene(args.worlds))
    exporter = FrameExporter(sim, worlds=sorted({0, args.worlds - 1}), max_points=20000)   # one world: [0]
    stop = threading.Event()
    thread = threading.Thread(target=publisher, args=(rr, exporter, stop), daemon=True)
    thread.start()
    for _ in range(int(round(args.seconds / sim.dt))):
        sim.step(1)
        exporter.capture()                              # never waits: a busy slot drops this frame
    exporter.flush()                                    # the last copy lands before the publisher's final poll
    stop.set()
    thread.join(timeout=10.0)
    print(f"stepped {sim.t:.2f} s; frames dropped {exporter.dropped}")


if __name__ == "__main__":
    main()
