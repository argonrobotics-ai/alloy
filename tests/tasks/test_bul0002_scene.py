# SPDX-License-Identifier: Apache-2.0
"""BUL-0002 regression on the converted scenes: Newton must keep the scene's convex hulls (no 64-vertex cut of its own;
T12's collision hulls are error-bounded pieces since the SB-6 split and BUL-0007), and the finding's trace must replay clean (no
S1) on CPU and on the GPU. See docs/notes/wp13-bul0002.md."""

from __future__ import annotations

import mujoco
from alloy._vendor import newton
import pytest

from alloy.bully.corpus import Corpus
from alloy.bully.replay import replay_trace
from alloy.bully.targets import get_target
from alloy.bully.types import BullyLimits
from alloy.tasks.mjcf_tree import hull_vertex_count
from alloy.tasks.replay import import_scene
from alloy.testing.tasks import REPO, ROBOTS


def _newton_hulls(scene: str) -> list[int]:
    model = import_scene(ROBOTS / scene / "scene.xml", "cpu")
    mj = newton.solvers.SolverMuJoCo(model, use_mujoco_contacts=True).mj_model
    return sorted(hull_vertex_count(mj, i) for i in range(mj.nmesh))


def _scene_collision_hulls(scene: str) -> list[int]:
    """Hull sizes of the distinct meshes the scene's colliding geoms use, compiled by MuJoCo from the MJCF."""
    m = mujoco.MjModel.from_xml_path(str(ROBOTS / scene / "scene.xml"))
    meshes = {int(m.geom_dataid[g]) for g in range(m.ngeom)
              if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and (m.geom_contype[g] or m.geom_conaffinity[g])}
    return sorted(hull_vertex_count(m, i) for i in meshes)


@pytest.mark.parametrize(("scene", "largest", "above_64"), [("task_t12", 62, 0), ("task_t11", 1423, 74)])
def test_newton_keeps_the_scene_hulls(scene: str, largest: int, above_64: int) -> None:
    """Newton builds every hull the scene asks for, with no 64-vertex cut of its own. Since BUL-0007, every T12
    collision hull is an error-bounded piece of at most 64 vertices (hull_split.json); before the split, T12 had 74
    hulls above 64 vertices, the largest with 1,423. T11 still carries the recorded hulls, so it keeps the check that
    Newton does not cut them."""
    got = _newton_hulls(scene)
    assert got == _scene_collision_hulls(scene)
    assert max(got) == largest and sum(h > 64 for h in got) == above_64


def _replay(device: str, runs: int):  # noqa: ANN202
    finding, trace, _, _ = Corpus(REPO / "bench" / "bully" / "corpus").load("BUL-0002")
    return replay_trace(get_target("task_t12"), trace, finding.monitor, device, BullyLimits(), runs=runs)


@pytest.mark.slow
@pytest.mark.timeout(900)
def test_bul0002_replays_clean_on_cpu() -> None:
    assert _replay("cpu", 3).trips == 0


@pytest.mark.gpu
@pytest.mark.timeout(1200)
def test_bul0002_replays_clean_on_gpu() -> None:
    import os

    assert _replay(os.environ.get("ALLOY_DEVICE", "cuda:0"), 10).trips == 0
