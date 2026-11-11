# SPDX-License-Identifier: Apache-2.0
"""FIX-LIST 7 law: ``alloy.estimate``'s wall time is within 1.3x of the measured run on its calibration rows.

Rows (``bench/results/estimate_calib/rows.json``; machines, commit and harness in its header): a robot carrying 40 ml of
water over a kinematic or a static cup, a bin of three shipped objects (and the same beside a kinematic paddle), the
shipped Panda scene, a 0.6 m sheet dropped on a table at each cloth setting and a 200 ml pour, at 1 / 4 / 64 worlds,
on one CPU core and on an L4. The measured wall is the warm steady run: on the L4 the periods that capture CUDA graphs
(about 6 after a build or reset, and about 6 more when a new graph key appears, such as SPH's own-CFL rung rising
mid-pour; 5-15x a replay period each) are left out, as the compile is.

Each row's scene is rebuilt here (the harness's own builders, [internal note] notes/wp2-estimate/rows.py). The L4 prediction
applies the cuda table to the template assembled on the CPU: the counts, the exchange clock and the load do not depend
on the device, and MuJoCo there is MuJoCo Warp. The CPU rows were measured on the users' default CPU route (MuJoCo C
where the route takes the entry), so the prediction asks for it too (the suite's conftest pins the Warp route).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import alloy
from alloy.api import _estimate as E

ROWS = Path(__file__).resolve().parents[2] / "bench" / "results" / "estimate_calib" / "rows.json"
BOUND = 1.3      # the coordinator's law is 2x; the docs say 1.3x, so the law holds that (worst row 1.24x at the fit)
ROBOT = """<mujoco><compiler angle="radian"/><worldbody>
  <body name="carriage" pos="0 0 0.25">
    <joint name="slide" type="slide" axis="1 0 0" range="-0.3 0.3" damping="50"/>
    <geom type="box" size="0.03 0.03 0.03" mass="2"/>
    <body name="wrist" pos="0 0 -0.06">
      <joint name="tilt" type="hinge" axis="0 1 0" range="-2.5 2.5" damping="5"/>
      <geom type="box" pos="0 0 -0.03" size="0.026 0.026 0.003" mass="0.05"/>
      <geom type="box" pos="0.024 0 -0.012" size="0.002 0.026 0.015" mass="0.02"/>
      <geom type="box" pos="-0.024 0 -0.012" size="0.002 0.026 0.015" mass="0.02"/>
      <geom type="box" pos="0 0.024 -0.012" size="0.026 0.002 0.015" mass="0.02"/>
      <geom type="box" pos="0 -0.024 -0.012" size="0.026 0.002 0.015" mass="0.02"/>
    </body>
  </body></worldbody>
  <actuator><position joint="slide" kp="2000" kv="200"/><position joint="tilt" kp="200" kv="20"/></actuator>
</mujoco>"""


def _robot_liquid(kinematic: bool) -> alloy.Scene:
    s = alloy.Scene(dt=0.02)
    s.add_ground()
    s.add_robot("arm", mjcf=ROBOT, pos=(0.0, 0.0, 0.0))
    s.add_liquid("water", size=(0.044, 0.044, 0.02), pos=(0.0, 0.0, 0.175))
    s.add_open_cylinder("cup", inner_radius=0.04, depth=0.08, wall=0.004, pos=(0.12, 0.0, 0.0),
                        **({"kinematic": True} if kinematic else {"static": True}))
    return s


def _bin(paddle: bool = False) -> alloy.Scene:
    s = alloy.Scene(dt=0.01)
    s.add_ground()
    s.add_open_box("bin", inner=(0.3, 0.3, 0.1), wall=0.01, pos=(0.0, 0.0, 0.0), static=True)
    s.add_object("mug", "mug_stoneware", pos=(-0.06, 0.0, 0.15))
    s.add_object("cube", "cube_beech_30", pos=(0.06, 0.05, 0.2))
    s.add_object("block", "block_alder_75x25x15", pos=(0.05, -0.07, 0.25))
    if paddle:
        s.add_box("paddle", size=(0.02, 0.2, 0.1), pos=(0.3, 0.0, 0.05), kinematic=True)
    return s


def _arm() -> alloy.Scene:
    from alloy.cards.library import default_root

    s = alloy.Scene(dt=0.01)
    s.add_robot("panda", default_root() / "robots" / "emb_e2_panda" / "scene.xml")
    return s


def _cloth(accuracy: str) -> alloy.Scene:
    s = alloy.Scene(dt=0.01, cloth_accuracy=accuracy)
    s.add_ground()
    s.add_box("table", size=(0.4, 0.4, 0.4), pos=(0.0, 0.0, 0.2), static=True)
    s.add_cloth("sheet", size=(0.6, 0.6), pos=(0.0, 0.0, 0.45))
    return s


def _pour() -> alloy.Scene:
    s = alloy.Scene(dt=0.005)
    s.add_ground()
    s.add_open_cylinder("cup", inner_radius=0.04, depth=0.10, wall=0.004, material="pp_homopolymer")
    s.add_pour("water", 200e-6, pos=(0.0, 0.0, 0.15), rate=2e-4)
    return s


SCENES = {"r1k": lambda: _robot_liquid(True), "r1s": lambda: _robot_liquid(False), "bin": _bin,
          "bin_k": lambda: _bin(True), "arm": _arm, "cloth_fast": lambda: _cloth("fast"),
          "cloth_default": lambda: _cloth("default"), "cloth_super": lambda: _cloth("super"), "pour": _pour}


def _rows() -> list[dict]:
    return json.loads(ROWS.read_text())["rows"]


@pytest.fixture(scope="module")
def templates():
    """Each row's one-world template (counts, solvers, steps per period, ..., load) on the CPU's default route."""
    mp = pytest.MonkeyPatch()
    mp.setenv("ALLOY_CPU_ROUTE", "mjc")
    try:
        yield {name: E._template_counts(make(), "cpu") for name, make in SCENES.items()}
    finally:
        mp.undo()


def _predicted(template: tuple, scene: alloy.Scene, row: dict) -> float:
    """The estimate's wall time for ``row`` (``_estimate._run_time``, as ``alloy.estimate``), with MuJoCo Warp on the L4."""
    kind = "cuda" if row["device"].startswith("cuda") else "cpu"
    load = template[6]
    if kind == "cuda":
        load = {**load, "mujoco": "warp" if load["mujoco"] else None}
    step_s, steps = E._run_time((*template[:6], load), kind, int(row["worlds"]), row["seconds"], scene.dt)
    return step_s * steps


def _worst(templates, key: str) -> tuple:
    """(factor off, measured / predicted, row, device, worlds) of the row furthest from its prediction in ``key``."""
    out = []
    for row in _rows():
        ratio = row[key] / _predicted(templates[row["row"]], SCENES[row["row"]](), row)
        out.append((max(ratio, 1.0 / ratio), ratio, row["row"], row["device"], row["worlds"]))
    return max(out)


def test_every_calibration_row_is_within_1p3x(templates):
    """measured / predicted within BOUND both ways for every row, CPU and L4 (the worst row named on failure)."""
    worst = _worst(templates, "measured_wall_s")
    assert worst[0] <= BOUND, f"{worst[2]} on {worst[3]} at {worst[4]} worlds: measured / predicted {worst[1]:.2f}"
    assert {(r["row"], r["device"].split(":")[0]) for r in _rows()} >= {(n, d) for n in SCENES for d in ("cpu", "cuda")}


def test_a_run_right_after_the_reset_is_within_3x(templates):
    """The same rows timed with their graph captures in (``first_run_wall_s``: the run right after ``sim.reset()``,
    kernels compiled): the estimate is of warm stepping, so a short GPU run after a build or reset is slower, up to
    2.65x on the 1 s robot + water rows (two graph keys, about 12 s of captures); the CPU captures nothing."""
    worst = _worst(templates, "first_run_wall_s")
    assert worst[0] <= 3.0, f"{worst[2]} on {worst[3]} at {worst[4]} worlds: measured / predicted {worst[1]:.2f}"


def test_the_estimate_keeps_its_keys():
    """OSS-3 keeps the API: the top-level keys and the counts' keys are the ones users read."""
    est = alloy.estimate(_bin(), worlds=2, seconds=0.1, device="cpu")
    assert {"physics_steps_per_world", "step_time_s", "wall_time_s", "world_steps_per_s", "real_time_factor",
            "memory_bytes", "compile_s_first_run", "counts", "solvers", "rigid_step", "routes", "vbd",
            "basis"} <= set(est)
    assert set(est["counts"]) == {"bodies", "dofs", "particles", "shapes", "mpm_particles"}
    assert np.isfinite(est["wall_time_s"]) and est["wall_time_s"] > 0


def test_a_pour_is_charged_its_particles_in_flow():
    """SPH's cost follows the particles that have left the nozzle (8x over a CPU pour from 12 % to all): over its
    own pouring time a pour averages half its particles, over twice that time three quarters."""
    assert E._active_share(1.0, 1.0) == pytest.approx(0.5)
    assert E._active_share(1.0, 2.0) == pytest.approx(0.75)
    assert E._active_share(1.0, 0.5) == pytest.approx(0.25)


MEMORY_ROWS = ROWS.with_name("memory_rows.json")
MEMORY_BOUND = 1.2      # what the docs say; the worst row is 1.19 (the one-hinge arm at 1,024 worlds, 0.84)
ARM1 = """<mujoco><compiler angle="radian"/><worldbody><body name="link1" pos="0 0 0.1">
  <joint name="shoulder" type="hinge" axis="0 1 0" range="-2 2" damping="2"/>
  <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.02" mass="1"/>
  </body></worldbody>
  <actuator><position joint="shoulder" kp="200" kv="10"/></actuator></mujoco>"""


def _arm1() -> alloy.Scene:
    """AGENTS.md section 5's one-hinge arm (the audit's arm)."""
    s = alloy.Scene(dt=0.01)
    s.add_ground()
    s.add_robot("arm", mjcf=ARM1, pos=(0, 0, 0.5))
    return s


def _ball() -> alloy.Scene:
    s = alloy.Scene(dt=0.01)
    s.add_ground()
    s.add_sphere("ball", 0.03, pos=(0.0, 0.0, 0.3), material="steel_carbon")
    return s


ARM2 = ARM1.replace("</body></worldbody>", """<body name="link2" pos="0.3 0 0">
  <joint name="elbow" type="hinge" axis="0 1 0" range="-2 2" damping="1"/>
  <geom type="capsule" fromto="0 0 0 0.25 0 0" size="0.02" mass="0.5"/></body>
  </body></worldbody>""").replace("</actuator>", '<position joint="elbow" kp="100" kv="5"/></actuator>')


def _arm2() -> alloy.Scene:
    s = alloy.Scene(dt=0.01)
    s.add_ground()
    s.add_robot("arm", mjcf=ARM2, pos=(0, 0, 0.5))
    return s


def _cloth_stand(flat: bool = False) -> alloy.Scene:
    """AGENTS.md section 7's sheet over a stand; ``flat``: on a table wider than it, without self-contact."""
    s = alloy.Scene(dt=0.01)
    s.add_ground()
    side = 0.8 if flat else 0.2
    s.add_box("stand", size=(side, side, 0.3), pos=(0, 0, 0.15), static=True)
    s.add_cloth("sheet", size=(0.5, 0.5), pos=(0, 0, 0.35), resolution=20, mass=0.1, self_contact=not flat)
    return s


def _liquid_tub() -> alloy.Scene:
    s = alloy.Scene(dt=0.01)
    s.add_ground()
    s.add_open_box("tub", inner=(0.10, 0.10, 0.08), wall=0.01, pos=(0, 0, 0))
    s.add_liquid("water", size=(0.09, 0.09, 0.04), pos=(0, 0, 0.03), material="water")
    return s


MEMORY_SCENES = {"arm1": _arm1, "ball": _ball, "arm2": _arm2, "cloth_stand": _cloth_stand,
                 "cloth_flat": lambda: _cloth_stand(flat=True), "liquid_tub": _liquid_tub}


def _memory_ratios(templates) -> list[tuple]:
    """(factor off, measured / predicted, row, worlds, source) of every row of ``memory_rows.json``."""
    tpl = {**templates, **{name: E._template_counts(make(), "cpu") for name, make in MEMORY_SCENES.items()}}
    out = []
    for row in json.loads(MEMORY_ROWS.read_text())["rows"]:
        ratio = row["measured_bytes"] / E._gpu_memory(tpl[row["row"]], int(row["worlds"]))
        out.append((max(ratio, 1.0 / ratio), ratio, row["row"], row["worlds"], row["source"]))
    return out


def test_gpu_memory_is_within_1p2x_on_every_memory_row(templates, monkeypatch):
    """Audits WORLDS-7 and WORLDS-8: the estimate read 289 MB for 16,384 worlds of a one-hinge arm that held 1,284 MB
    on an L4, and 254 MB for 1,024 worlds of AGENTS.md's cloth scene that held 2,492 MB (it sized 16 contacts and 64
    rows per world where the build allocates 64 and 384, counted 400 B a particle, and left out the scratch a CUDA
    graph holds and the CUDA context). ``memory_rows.json``, one L4 each: WP2's 20 rows (arm, ball, bin, Panda
    scene, cloth, pour, robot with water; 1 to 16,384 worlds) and the audit's own 18 (its harness and tree: cloth
    over a stand and flat, water in a tub, a two-hinge arm; 1 to 4,096 worlds). measured / predicted within
    MEMORY_BOUND both ways on all of them."""
    monkeypatch.setenv("ALLOY_CPU_ROUTE", "mjc")
    rows = _memory_ratios(templates)
    worst = max(rows)
    assert worst[0] <= MEMORY_BOUND, f"{worst[2]} at {worst[3]} worlds ({worst[4]}): measured / predicted {worst[1]:.2f}"
    assert {(r[2], r[3]) for r in rows} >= {("arm1", 16384), ("bin", 4096), ("arm", 256), ("cloth_stand", 1024),
                                            ("cloth_flat", 1024), ("liquid_tub", 1024)}
    assert len([r for r in rows if r[4] == "audit"]) >= 18
    tpl = E._template_counts(_arm1(), "cpu")
    audit = next(r for r in json.loads(MEMORY_ROWS.read_text())["rows"] if (r["row"], r["worlds"]) == ("arm1", 16384))
    assert audit["measured_bytes"] / E._memory(tpl[0], 16384) > 4.0               # the model before: 4-5x low


def test_a_mujoco_entry_without_sizes_gets_the_builds_floors():
    """OSS-3's review: the memory model reads contact slots and rows from the entries; one that names none gets the
    build's default rule (at least 64 slots and 384 rows a world), not zero."""
    from types import SimpleNamespace

    bare, sized = SimpleNamespace(solver_kwargs={}), SimpleNamespace(solver_kwargs={"nconmax": 100, "njmax": 500})
    assert E._capacities([bare], 2) == {"nconmax": 64, "njmax": 384}
    assert E._capacities([bare], 20) == {"nconmax": 160, "njmax": 960}
    assert E._capacities([sized, bare], 2) == {"nconmax": 164, "njmax": 884}
    assert E._capacities([], 2) == {"nconmax": 0, "njmax": 0}
