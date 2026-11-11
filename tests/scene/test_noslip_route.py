# SPDX-License-Identifier: Apache-2.0
"""SCR-WP3-4: a scene whose robot MJCF sets noslip_iterations > 0 runs its MuJoCo entry with noslip (scoped to the
robot on MuJoCo Warp with an elliptic cone; MuJoCo's own with a pyramidal cone; none under `discrete`)."""

from __future__ import annotations

import numpy as np
import pytest

from alloy.cards.types import Placement, RobotRef, SceneCard
from alloy.core.registry import isolated_registry
from alloy.scene import noslip_route as NR
from alloy.scene.assembler import SceneAssembler
from alloy.sim import Simulation

UP = (0.0, 0.0, 0.0, 1.0)
YAM = "robots/yam_bench/robot.xml"


def _lib():
    try:
        from alloy.cards.library import CardLibrary
    except ImportError:
        pytest.skip("WP1 CardLibrary not merged")
    lib = CardLibrary()
    if lib.asset_path(YAM) is None or not lib.asset_path(YAM).exists():
        pytest.skip("WP13 bench MJCF not vendored")
    return lib


def test_mjcf_noslip_reads_the_recorded_option():
    lib = _lib()
    req = NR.mjcf_noslip(str(lib.asset_path(YAM)))
    assert req is not None and (req.iterations, req.tolerance) == (50, 1e-10)
    bin50 = lib.asset_path("robots/task_bin50/scene.xml")
    if bin50 is not None and bin50.exists():
        assert NR.mjcf_noslip(str(bin50)) is None                          # noslip_iterations="0"


def _built(lib, worlds: int = 2):
    robot = RobotRef(path=YAM, kind="mjcf", base_pose=(0.0, 0.0, 0.0, *UP), name="yam_bench")
    card = SceneCard(robots=(robot,), objects=(Placement("cube_beech_30", (0.35, 0.15, 0.0155, *UP)),), worlds=worlds)
    return SceneAssembler(lib).assemble(card, "cpu", world_count=worlds)


@pytest.mark.compile_heavy
@pytest.mark.timeout(600)
def test_routed_robot_scene_installs_scoped_noslip():
    from alloy.contact import noslip as NS

    lib = _lib()
    with isolated_registry(empty=True):
        built = _built(lib)
        (mj,) = [e for e in built.entries if e.capability == "mujoco"]
        assert mj.solver_kwargs[NR.NOSLIP_KEY] == (50, 1e-10)
        assert "mujoco:noslip=50" in built.report.scene_effects
        sim = Simulation(built)
        solver = sim.stepper.solvers()[0]
        assert int(solver.mjw_model.opt.noslip_iterations) == 50
        assert id(solver.mjw_model) in NS._ROUTES                       # MuJoCo Warp's noslip call goes to the scoped one
        for _ in range(2):
            sim.step_period()
        assert np.isfinite(sim.state.body_q.numpy()).all()


@pytest.mark.compile_heavy
@pytest.mark.timeout(600)
@pytest.mark.parametrize("kw,mode", [({"cone": "pyramidal"}, "mujoco"), ({"integrator": "discrete"}, "off"),
                                     ({"use_mujoco_cpu": True}, "mujoco")])
def test_noslip_mode_follows_cone_and_integrator(kw, mode):
    from alloy._vendor import newton

    lib = _lib()
    with isolated_registry(empty=True):
        built = _built(lib, worlds=1)
        (mj,) = [e for e in built.entries if e.capability == "mujoco"]
        args = {k: v for k, v in mj.solver_kwargs.items() if k != NR.NOSLIP_KEY} | kw
        solver = newton.solvers.SolverMuJoCo(built.model, **args)
        assert NR.install_entry_noslip(solver, (50, 1e-10)) == mode
        want = 50 if mode == "mujoco" else 0
        assert int(solver.mj_model.opt.noslip_iterations) == want


# SCR-SC6-3: noslip from every MJCF the scene loads (robots and MJCF object cards: benches, fixtures).

def _mjcf(tmp_path, name: str, noslip: tuple[int, float] | None = None) -> str:
    opt = f'<option noslip_iterations="{noslip[0]}" noslip_tolerance="{noslip[1]}"/>' if noslip else ""
    path = tmp_path / name
    path.write_text(f"<mujoco>{opt}<worldbody/></mujoco>")
    return str(path)


def _mjcf_object(path: str):
    from types import SimpleNamespace as NS

    from alloy.cards.types import MJCFSource

    return NS(obj=NS(card=NS(shape=MJCFSource(path=path))))


def test_scene_noslip_reads_mjcf_objects_too(tmp_path):
    robot = RobotRef(path="robot.xml", kind="mjcf", base_pose=(0.0, 0.0, 0.0, *UP), name="r")
    files = {"robot.xml": _mjcf(tmp_path, "robot.xml")}                        # the robot asks for none
    bench = _mjcf(tmp_path, "bench.xml", (50, 1e-10))
    req = NR.scene_noslip((robot,), files.__getitem__, [_mjcf_object(bench)])
    assert (req.iterations, req.tolerance, req.source) == (50, 1e-10, bench)
    assert "noslip_iterations=50" in NR.report_line(req) and "/bench.xml" in NR.report_line(req)
    assert NR.scene_noslip((robot,), files.__getitem__, [_mjcf_object(_mjcf(tmp_path, "plain.xml"))]) is None
    files["robot.xml"] = _mjcf(tmp_path, "robot.xml", (50, 1e-6))             # most iterations, then tightest tol
    more = _mjcf(tmp_path, "more.xml", (60, 1e-6))
    assert NR.scene_noslip((robot,), files.__getitem__, [_mjcf_object(bench)]).tolerance == 1e-10
    assert NR.scene_noslip((robot,), files.__getitem__, [_mjcf_object(bench), _mjcf_object(more)]).iterations == 60


BENCH = "box_blank_sc4"                       # an MJCF object card (mjcf/clips/sc4/box_blank_sc4.xml)


@pytest.fixture(scope="module")
def bench_assets(tmp_path_factory):
    """A private alloy-assets copy whose MJCF bench object asks for noslip 50 / 1e-10 (the shared checkout is
    never written)."""
    import shutil

    from alloy.cards.roots import assets_root

    src = assets_root()
    if src is None or not (src / "mjcf/clips/sc4/box_blank_sc4.xml").exists():
        pytest.skip("alloy-assets checkout with the SC4 bench MJCF not found")
    dst = tmp_path_factory.mktemp("assets") / "alloy-assets"
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns(".git"))
    xml = dst / "mjcf/clips/sc4/box_blank_sc4.xml"
    text = xml.read_text()
    head = text.index(">", text.index("<mujoco")) + 1
    xml.write_text(text[:head] + '<option noslip_iterations="50" noslip_tolerance="1e-10"/>' + text[head:])
    return dst


def _bench_scene(assets=None):
    from alloy.cards.library import CardLibrary
    from alloy.cards.types import Placement, SceneCard

    lib = CardLibrary() if assets is None else CardLibrary(assets=assets)
    card = SceneCard(robots=(), objects=(Placement(BENCH, (0.0, 0.0, 0.05, *UP)),), worlds=1)
    return SceneAssembler(lib).assemble(card, "cpu", world_count=1)


def test_bench_mjcf_noslip_reaches_the_solver(bench_assets):
    from alloy._vendor import newton

    with isolated_registry(empty=True):
        built = _bench_scene(bench_assets)
        (mj,) = [e for e in built.entries if e.capability == "mujoco"]
        assert mj.solver_kwargs[NR.NOSLIP_KEY] == (50, 1e-10)
        assert "mujoco:noslip=50" in built.report.scene_effects
        assert any("noslip_iterations=50" in x and "sc4/box_blank_sc4.xml" in x for x in built.report.entry_options)
        solver = Simulation(built).stepper.solvers()[0]
        assert isinstance(solver, newton.solvers.SolverMuJoCo)
        assert int(solver.mjw_model.opt.noslip_iterations) == 50


def test_no_mjcf_noslip_gives_none(bench_assets, tmp_path):
    from alloy._vendor import newton

    with isolated_registry(empty=True):
        built = _bench_scene()                                  # the shared checkout's bench asks for none
        (mj,) = [e for e in built.entries if e.capability == "mujoco"]
        assert NR.NOSLIP_KEY not in mj.solver_kwargs
        assert not any("noslip" in x for x in built.report.entry_options)
        solver = newton.solvers.SolverMuJoCo(built.model, use_mujoco_contacts=True, cone="elliptic")
        assert NR.install_file_noslip(solver, [_mjcf(tmp_path, "plain.xml")]) == "none"
        assert int(solver.mjw_model.opt.noslip_iterations) == 0
        # a directly built solver gets the noslip of the files it loaded, by the route's rule
        bench = str(bench_assets / "mjcf/clips/sc4/box_blank_sc4.xml")
        assert NR.install_file_noslip(solver, [bench]) == "scoped"
        assert int(solver.mjw_model.opt.noslip_iterations) == 50
