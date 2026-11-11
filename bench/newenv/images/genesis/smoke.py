# SPDX-License-Identifier: Apache-2.0
"""Genesis 1.4.6 smoke: 16 parallel envs of a falling box + Franka, 100 steps; then one PBD cloth and one SPH liquid
scene (build, 10 steps) so the Quadrants kernel cache holds the soft-matter solvers too. One JSON line."""
import json
import time

import genesis as gs
import torch

N, STEPS = 16, 100
gs.init(backend=gs.gpu, logging_level="warning")


def rigid() -> dict:
    scene = gs.Scene(show_viewer=False)
    scene.add_entity(gs.morphs.Plane())
    scene.add_entity(gs.morphs.Box(size=(0.1, 0.1, 0.1), pos=(0, 0, 0.5)))
    scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))
    t0 = time.perf_counter()
    scene.build(n_envs=N)
    t_build = time.perf_counter() - t0
    scene.step()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(STEPS):
        scene.step()
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    return {"steps_per_s": round(STEPS / wall, 1), "env_steps_per_s": round(N * STEPS / wall), "t_build_s": round(t_build, 1)}


def cloth() -> float:
    scene = gs.Scene(sim_options=gs.options.SimOptions(dt=4e-3, substeps=10), show_viewer=False)
    scene.add_entity(gs.morphs.Plane())
    scene.add_entity(material=gs.materials.PBD.Cloth(), morph=gs.morphs.Mesh(file="meshes/cloth.obj", scale=1.0, pos=(0, 0, 0.5)))
    t0 = time.perf_counter()
    scene.build(n_envs=N)
    for _ in range(10):
        scene.step()
    return round(time.perf_counter() - t0, 1)


def liquid() -> float:
    scene = gs.Scene(sim_options=gs.options.SimOptions(dt=4e-3, substeps=10),
                     sph_options=gs.options.SPHOptions(lower_bound=(-0.5, -0.5, 0.0), upper_bound=(0.5, 0.5, 1.0), particle_size=0.01),
                     show_viewer=False)
    scene.add_entity(gs.morphs.Plane())
    scene.add_entity(material=gs.materials.SPH.Liquid(), morph=gs.morphs.Box(pos=(0, 0, 0.2), size=(0.2, 0.2, 0.2)))
    t0 = time.perf_counter()
    scene.build(n_envs=N)
    for _ in range(10):
        scene.step()
    return round(time.perf_counter() - t0, 1)


if __name__ == "__main__":
    out = {"arm": "genesis", "envs": N, "steps": STEPS, **rigid()}
    out["cloth_build_plus_10_steps_s"] = cloth()
    out["liquid_build_plus_10_steps_s"] = liquid()
    print(json.dumps(out))
