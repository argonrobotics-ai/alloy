# SPDX-License-Identifier: Apache-2.0
"""Convex and decomposed meshes beside MPM material (audit S1-007; vendored patch 0016-mpm-convex-mesh-collider).

Before the patch every ``add_mesh(collision="convex" | "decompose")`` body beside sand or paste crashed the build with
Newton's bare ``NotImplementedError: Shape type 10 not supported``: its MPM collider setup meshed primitives and
triangle meshes but not convex-mesh shapes. Now a convex-mesh shape's MPM collider is the convex hull of its source
vertices, the shape rigid contact uses. CPU laws:
- the audit's four builds (convex / decompose, beside sand / mayonnaise) succeed;
- sand dropped on a free open cylinder with ``collision="convex"`` rests on the hull's top (the rim plane), never
  more than 1 mm inside the hull;
- with ``collision="decompose"`` the same sand falls into the cavity and rests on the floor part, at most 4 of
  2,048 grains more than 1 mm inside any part (2 measured, the deepest 1.16 mm);
- a shape MPM still cannot mesh (an MJCF ellipsoid beside sand) is a BuildError with a fix, not a bare
  NotImplementedError.

    PYTHONPATH=. python -m pytest tests/integration/test_mpm_convex_mesh.py -q
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial import ConvexHull

import alloy
from alloy.api import meshes as M

pytestmark = pytest.mark.integration

CONVEX_MESH = 10                       # Newton's GeoType.CONVEX_MESH
WALL, DEPTH, INNER = 0.006, 0.03, 0.03
RIM = DEPTH + WALL                     # the cylinder's top, its hull's top face (m)
INSIDE = 1.0e-3                        # a grain "inside" a hull: deeper than this (m)
ELLIPSOID_ARM = """<mujoco><compiler angle="radian"/><worldbody><body name="link1" pos="0 0 0.1">
  <joint name="shoulder" type="hinge" axis="0 1 0" range="-2 2" damping="2"/>
  <geom type="ellipsoid" size="0.02 0.01 0.01" mass="0.2"/>
  </body></worldbody>
  <actuator><position joint="shoulder" kp="20" kv="1"/></actuator></mujoco>"""


def _needs_decompose_extra() -> None:
    pytest.importorskip("coacd", reason="collision='decompose' needs the import extra")


def _rot(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    u, w = q[..., :3], q[..., 3:4]
    t = 2.0 * np.cross(u, v)
    return v + w * t + np.cross(u, t)


def _inv(q: np.ndarray) -> np.ndarray:
    return np.concatenate([-q[:3], q[3:]])


def hull_depths(sim: alloy.Sim, pts: np.ndarray) -> np.ndarray:
    """Per grain, how deep it is inside the deepest convex-mesh hull of the scene (m; negative outside)."""
    m, st = sim.engine.model, sim.engine.state
    kind, body, xf = m.shape_type.numpy(), m.shape_body.numpy(), m.shape_transform.numpy()
    bq, scale = st.body_q.numpy(), m.shape_scale.numpy()
    out = np.full(len(pts), -np.inf)
    for k in np.flatnonzero(kind == CONVEX_MESH):
        eq = ConvexHull(np.asarray(m.shape_source[k].vertices, float) * scale[k]).equations
        b = bq[body[k]] if body[k] >= 0 else np.array([0.0, 0, 0, 0, 0, 0, 1])
        origin = b[:3] + _rot(b[3:7], xf[k, :3])
        local = _rot(_inv(xf[k, 3:7])[None], _rot(_inv(b[3:7])[None], pts - origin[None]))
        out = np.maximum(out, -(local @ eq[:, :3].T + eq[:, 3]).max(axis=1))
    return out


def _bowl_scene(collision: str, material: str = "sand_silica_dry", drop: bool = True) -> alloy.Scene:
    v, f = M.open_cylinder(inner_radius=INNER, depth=DEPTH, wall=WALL, segments=12)
    s = alloy.Scene(dt=0.01)
    s.add_ground()
    s.add_mesh("bowl", vertices=v, faces=f, pos=(0.0, 0.0, 0.0), collision=collision, mass=0.8)
    at = (0.0, 0.0, RIM + 0.015) if drop else (-0.30, 0.0, 0.04)
    s.add_box("sand", size=(0.02, 0.02, 0.01), pos=at, material=material)
    return s


def _drop(collision: str, seconds: float = 0.3) -> tuple[np.ndarray, float, np.ndarray, bool]:
    """Sand dropped 1 cm onto the bowl: (grains over the bowl at the end, the deepest any grain went into a hull at
    any control period, the end's per-grain hull depths, sim.check()["ok"])."""
    sim = alloy.build(_bowl_scene(collision), device="cpu", seed=0)
    deepest = -np.inf
    for _ in range(int(round(seconds / 0.01))):
        sim.step(1)
        pts = np.asarray(sim.state(["sand"])["sand.points"])[0].astype(float)
        deepest = max(deepest, float(hull_depths(sim, pts).max()))
    pts = np.asarray(sim.state(["sand"])["sand.points"])[0].astype(float)
    centre = np.asarray(sim.state(["bowl"])["bowl.pos"])[0]
    over = pts[np.linalg.norm(pts[:, :2] - centre[None, :2], axis=1) < INNER + WALL]
    return over, deepest, hull_depths(sim, pts), bool(sim.check()["ok"])


@pytest.mark.parametrize("material", ["sand_silica_dry", "mayonnaise"])
@pytest.mark.parametrize("collision", ["convex", "decompose"])
def test_convex_and_decomposed_meshes_build_beside_sand_and_paste(collision: str, material: str) -> None:
    """The audit's repro: before, all four raised NotImplementedError('Shape type 10 not supported')."""
    if collision == "decompose":
        _needs_decompose_extra()
    sim = alloy.build(_bowl_scene(collision, material, drop=False), device="cpu", seed=0)
    assert "mpm" in {r["route"].split("_")[-1] for r in sim.routes().values()}
    sim.close()


@pytest.mark.timeout(600)                # about 100 s of CPU MPM each
def test_sand_rests_on_a_convex_mesh_hull() -> None:
    """A convex mesh collides as its hull, so the sand stays on the rim plane and never enters the cavity."""
    over, deepest, _, ok = _drop("convex")
    assert ok
    assert len(over) >= 2000                                       # 2,044 of 2,048 measured; 4 scatter off
    assert np.percentile(over[:, 2], 1) >= RIM - INSIDE, np.percentile(over[:, 2], 1)    # 36.27 mm measured
    assert np.percentile(over[:, 2], 50) <= RIM + 0.005                                  # resting on it: 37.3 mm
    assert deepest <= INSIDE, deepest                              # 0.20 mm measured


@pytest.mark.timeout(600)                # about 100 s of CPU MPM each
def test_sand_fills_a_decomposed_bowl_and_rests_on_its_floor() -> None:
    """The decomposed parts keep the cavity open: the sand falls in and rests on the floor part (top at z = WALL)."""
    _needs_decompose_extra()
    over, deepest, end, ok = _drop("decompose")
    assert ok
    assert len(over) >= 2000
    assert np.percentile(over[:, 2], 1) >= WALL - INSIDE, np.percentile(over[:, 2], 1)   # 6.28 mm measured
    assert np.percentile(over[:, 2], 50) <= WALL + 0.005                                 # in the cavity: 6.94 mm
    assert int((end > INSIDE).sum()) <= 4 and deepest <= 2 * INSIDE, (int((end > INSIDE).sum()), deepest)


def test_a_shape_mpm_cannot_mesh_is_a_build_error_with_a_fix() -> None:
    s = alloy.Scene(dt=0.01)
    s.add_ground()
    s.add_box("sand", size=(0.03, 0.03, 0.01), pos=(0.05, 0, 0.005), material="sand_silica_dry")
    s.add_robot("arm", mjcf=ELLIPSOID_ARM, pos=(0, 0, 0.0))
    with pytest.raises(alloy.BuildError) as caught:
        alloy.build(s, device="cpu")
    assert "ellipsoid" in str(caught.value) and "collision=" in caught.value.fix
