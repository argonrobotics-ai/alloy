# SPDX-License-Identifier: Apache-2.0
"""Rough ground for the Go2: the terrain grid (no simulation), its MuJoCo heightfield file, the level curriculum and
the off-field ending on a 9-world CPU environment, and the flat environment's heights left as they were."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest

from alloy.cards.library import default_root
from alloy.envs import EnvError
from alloy.envs.backend import NumpyBackend
from alloy.envs.loco import Go2VelocityConfig, go2_velocity
from alloy.envs.loco.env import stand_action
from alloy.envs.loco.rough import Go2RoughConfig, go2_rough
from alloy.envs.loco.terrain import Terrain, TerrainSpec, terrain_mjcf
from alloy.envs.mjcf import without_bodies

W, LEVELS, TILE, SECOND = 9, 3, 4.0, 50          # worlds, levels, tile edge (m), control steps in 1 s (dt 0.02)
SPEC = TerrainSpec(border=0.0, levels=LEVELS, tile=TILE)
GO2_FILE = default_root() / "robots" / "emb_e8_go2" / "scene.xml"
XP = NumpyBackend("cpu")


def _per_tile(t: Terrain) -> int:
    return int(round(t.spec.tile / t.cell))


def _tile(t: Terrain, level: int, kind: int) -> np.ndarray:
    p = _per_tile(t)
    return t.heights[kind * p:(kind + 1) * p + 1, level * p:(level + 1) * p + 1]


@pytest.fixture(scope="module")
def terrain() -> Terrain:
    return Terrain(TerrainSpec(border=0.0, levels=5, tile=TILE))


# --- A. the terrain alone -------------------------------------------------------------------------------------------

def test_grid_shape_is_nodes_of_kinds_by_levels(terrain: Terrain) -> None:
    """The grid has kinds * 40 + 1 rows and levels * 40 + 1 columns at a 0.1 m cell and 4 m tiles."""
    assert terrain.heights.shape == (3 * 40 + 1, 5 * 40 + 1)


def test_level_zero_is_exactly_flat(terrain: Terrain) -> None:
    """Every tile of level 0 is at height 0 exactly."""
    assert all((_tile(terrain, 0, k) == 0.0).all() for k in range(terrain.kinds))


def test_field_and_tile_borders_are_at_zero(terrain: Terrain) -> None:
    """The field's border and every tile's border are at height 0 (within 1e-6 m)."""
    h, p = terrain.heights, _per_tile(terrain)
    assert np.abs(h[[0, -1], :]).max() < 1e-6 and np.abs(h[:, [0, -1]]).max() < 1e-6
    assert np.abs(h[::p, :]).max() < 1e-6 and np.abs(h[:, ::p]).max() < 1e-6


def test_platforms_are_flat_at_zero(terrain: Terrain) -> None:
    """Within platform / 2 of each tile's centre the ground is at height 0 (within 1e-6 m) at every level."""
    p, half = _per_tile(terrain), 0.5 * terrain.spec.platform
    u = (np.arange(p + 1) - 0.5 * p) * terrain.cell
    on = np.maximum(np.abs(u)[:, None], np.abs(u)[None, :]) <= half + 1e-9
    assert on.sum() > 100
    for level in range(terrain.levels):
        for kind in range(terrain.kinds):
            assert np.abs(_tile(terrain, level, kind)[on]).max() < 1e-6


def _level_maxima(t: Terrain, kind: int) -> np.ndarray:
    return np.array([_tile(t, level, kind).max() for level in range(t.levels)])


def test_waves_reach_their_height_at_the_last_level(terrain: Terrain) -> None:
    """The crest of the waves at the last level is 0.15 m within 5 %, and the crests grow with the level."""
    peaks = _level_maxima(terrain, 1)
    assert peaks[-1] == pytest.approx(terrain.spec.wave_height, rel=0.05)
    assert (np.diff(peaks) > 0).all()


@pytest.mark.parametrize(("kind", "limit"), [(0, 0.06), (2, 0.08)])
def test_rough_and_blocks_stay_within_their_height(terrain: Terrain, kind: int, limit: float) -> None:
    """Rough ground peaks at <= 0.06 m and blocks at <= 0.08 m at the last level, over 60 % of it, and grow."""
    peaks = _level_maxima(terrain, kind)
    assert 0.6 * limit < peaks[-1] <= limit
    assert (np.diff(peaks) > 0).all()


def test_seed_decides_the_heights() -> None:
    """The same seed gives identical heights; another seed changes them by more than 1 mm somewhere."""
    a, b, c = (Terrain(TerrainSpec(border=0.0, levels=3, tile=TILE, seed=s)).heights for s in (4, 4, 5))
    np.testing.assert_array_equal(a, b)
    assert np.abs(a - c).max() > 1e-3


def test_height_at_is_the_grid_at_nodes_and_the_mean_between(terrain: Terrain) -> None:
    """height_at equals the grid at nodes and the mean of two neighbours midway (x and y), within 1e-5 m."""
    p = _per_tile(terrain)
    j, i = np.meshgrid(np.arange(2 * p, 2 * p + 8), np.arange(4 * p + 5, 4 * p + 12), indexing="ij")
    j, i = j.ravel(), i.ravel()
    x, y = terrain.x0 + i * terrain.cell, terrain.y0 + j * terrain.cell
    h = terrain.heights
    assert np.abs(h[j, i]).max() > 0.01                                   # a bumpy patch (rough, last level)
    np.testing.assert_allclose(terrain.height_at(XP, np.stack([x, y], -1)), h[j, i], atol=1e-5)
    mid_x = terrain.height_at(XP, np.stack([x + 0.5 * terrain.cell, y], -1))
    mid_y = terrain.height_at(XP, np.stack([x, y + 0.5 * terrain.cell], -1))
    np.testing.assert_allclose(mid_x, 0.5 * (h[j, i] + h[j, i + 1]), atol=1e-5)
    np.testing.assert_allclose(mid_y, 0.5 * (h[j, i] + h[j + 1, i]), atol=1e-5)


def test_middle_tile_is_centred_on_the_origin(terrain: Terrain) -> None:
    """The middle tile of the 5 x 3 field is centred at (0, 0) and the end tiles 8 m away along x."""
    assert terrain.tile_center(2, 1) == (0.0, 0.0)
    assert terrain.tile_center(0, 1) == (-8.0, 0.0) and terrain.tile_center(4, 1) == (8.0, 0.0)


def test_inside_is_false_beyond_the_field_edge(terrain: Terrain) -> None:
    """Points beyond 10 m in x or 6 m in y are outside; the origin and a point 0.5 m from the edge are inside."""
    xy = np.array([[0.0, 0.0], [9.5, 0.0], [10.5, 0.0], [0.0, 6.5], [-10.5, -6.5]])
    assert terrain.inside(xy).tolist() == [True, True, False, False, False]


@pytest.mark.parametrize("spec", [TerrainSpec(border=0.0, kinds=("rough", "ice")), TerrainSpec(border=0.0, levels=4),
                                  TerrainSpec(border=0.0, tile=1.0, platform=1.0)],
                         ids=["kind", "even levels", "platform"])
def test_bad_specs_raise(spec: TerrainSpec) -> None:
    """An unknown kind, an even number of levels and a platform as wide as the tile raise EnvError."""
    with pytest.raises(EnvError):
        spec.validate()


# --- B. the MuJoCo file ---------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def mj_field() -> tuple[Terrain, Path, mujoco.MjModel, Path]:
    terrain = Terrain(SPEC)
    src = without_bodies(GO2_FILE, ["obstacle"])
    out = terrain_mjcf(src, terrain)
    return terrain, src, mujoco.MjModel.from_xml_path(str(out)), out


def _surface_z(model: mujoco.MjModel, data: mujoco.MjData, x: float, y: float) -> float:
    geom = np.zeros(1, np.int32)
    only_terrain = np.array([0, 0, 0, 0, 0, 1], np.uint8)          # the field is moved to group 5 (not the robot's)
    dist = mujoco.mj_ray(model, data, np.array([x, y, 2.0]), np.array([0.0, 0.0, -1.0]), only_terrain, 1, -1, geom)
    assert dist > 0 and mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(geom[0])) == "terrain"
    return 2.0 - dist


def test_heightfield_surface_matches_height_at(mj_field) -> None:
    """A ray down at a platform centre and 5 bump nodes (rough, waves, blocks; levels 1 and 2) hits the field
    within 2 mm of Terrain.height_at there."""
    terrain, _, model, _ = mj_field
    model.geom_group[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "terrain")] = 5
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    p = _per_tile(terrain)
    picks = []
    for level, kind in [(2, 0), (2, 1), (2, 2), (1, 0), (1, 1)]:
        tile = _tile(terrain, level, kind)
        j, i = np.unravel_index(int(np.argmax(tile)), tile.shape)
        picks.append((terrain.x0 + (level * p + i) * terrain.cell, terrain.y0 + (kind * p + j) * terrain.cell))
    picks.insert(0, terrain.tile_center(1, 1))
    heights = [float(terrain.height_at(XP, np.array(pt))) for pt in picks]
    assert sum(h > 0.01 for h in heights) == 5
    for (x, y), expected in zip(picks, heights, strict=True):
        assert _surface_z(model, data, x, y) == pytest.approx(expected, abs=0.002)


def test_floor_plane_is_replaced_by_a_terrain_with_its_friction(mj_field) -> None:
    """The plane 'floor' is gone and the 'terrain' geom has the plane's friction."""
    _, src, model, _ = mj_field
    plain = mujoco.MjModel.from_xml_path(str(src))
    floor = mujoco.mj_name2id(plain, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    assert floor >= 0
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor") == -1
    ground = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "terrain")
    np.testing.assert_array_equal(model.geom_friction[ground], plain.geom_friction[floor])
    assert model.geom_type[ground] == mujoco.mjtGeom.mjGEOM_HFIELD


def test_file_without_a_floor_plane_raises(tmp_path: Path) -> None:
    """A robot file with no plane named 'floor' raises EnvError."""
    xml = tmp_path / "bare.xml"
    xml.write_text('<mujoco><worldbody><geom name="ground" type="plane" size="1 1 0.1"/></worldbody></mujoco>')
    with pytest.raises(EnvError, match="floor"):
        terrain_mjcf(xml, Terrain(SPEC))


# --- C. the environment ---------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def env():
    config = Go2RoughConfig(pushes=None, start_levels=LEVELS, terrain=SPEC, contact="force")
    e = go2_rough(num_envs=W, device="cpu", backend="numpy", config=config)
    e.initial_levels = np.array(e.levels).copy()
    yield e
    e.close()


def _restart(env, curriculum: bool = False) -> None:
    """Back to the dealt levels on their platforms, a fresh random command, curriculum as asked."""
    env.curriculum = False
    env.commands.fixed = False
    env.levels = env.xp.asarray(env.initial_levels, "int64")
    env.reset(seed=0)
    env.curriculum = curriculum


def _stand(env, steps: int = SECOND) -> None:
    for _ in range(steps):
        env.step(stand_action(env))


def _tiles(env) -> np.ndarray:
    x, y = env.terrain.tile_center(np.array(env.levels, np.float32), np.array(env.kinds, np.float32))
    return np.stack([x, y], -1)


def _base(env) -> np.ndarray:
    return np.array(env.state["go2/base.pos"])


def _weight(env) -> np.ndarray:
    return env.sim.engine.model.body_mass.numpy().reshape(W, -1).sum(1) * 9.81


def _assert_standing(env, worlds: np.ndarray) -> None:
    on = _base(env)
    assert np.abs(on[worlds, 0:2] - _tiles(env)[worlds]).max() < 0.1
    assert np.abs(np.array(env.state["base.height"])[worlds] - 0.264).max() < 0.01
    force, weight = np.array(env.state["feet.force"]).sum(1), _weight(env)
    np.testing.assert_allclose(force[worlds], weight[worlds], rtol=0.03, atol=0)
    feet = np.array(env.state["feet.height"])[worlds]
    assert feet.min() >= -0.002 and feet.max() <= 0.005


@pytest.mark.compile_heavy
def test_robots_stand_on_their_own_tiles(env) -> None:
    """After 1 s of zero actions all 9 robots stand on their tile: base within 0.1 m of its centre, 0.264 m high,
    feet carry the weight within 3 %, soles within [-2, 5] mm, health ok."""
    _restart(env)
    assert len(set(zip(np.array(env.levels).tolist(), np.array(env.kinds).tolist(), strict=True))) == W
    _stand(env)
    _assert_standing(env, np.arange(W))
    assert env.sim.check()["ok"]


@pytest.mark.compile_heavy
def test_level_is_a_state_key_and_obs_widths_are_the_flat_ones(env) -> None:
    """env.state['terrain.level'] equals env.levels; the observation is 55 wide (policy) and 79 (critic)."""
    _restart(env)
    np.testing.assert_array_equal(np.array(env.state["terrain.level"]), np.array(env.levels))
    obs, _ = env.reset(seed=0)
    assert obs["policy"].shape == (W, 55) and obs["critic"].shape == (W, 79)


# --- D. the curriculum ----------------------------------------------------------------------------------------------

def _teleport(env, moves: dict[int, float]) -> None:
    """Move each world's robot ``moves[w]`` metres in x (toward the middle) at 0.30 m above the ground there."""
    pos = _base(env).copy()
    for w, d in moves.items():
        sign = -1.0 if pos[w, 0] > 0 else 1.0
        pos[w, 0] += sign * d
        pos[w, 2] = float(env.terrain.height_at(env.xp, env.xp.asarray(pos[w:w + 1, 0:2]))[0]) + 0.30
    env.sim.set_pose("go2", pos=pos.astype(np.float32))


@pytest.fixture(scope="module")
def sweep(env) -> dict:
    """1 s standing at 1 m/s commanded, then robots moved by hand: 0 and 6 far, 2 and 7 a little (1.0 m), 1, 3 and 4
    not at all; worlds 0-7 are reset, 8 (moved far) is not; 1 s of standing after the reset."""
    _restart(env, curriculum=True)
    env.commands.set((1.0, 0.0, 0.0))
    _stand(env)
    assert int(np.array(env.state["steps"]).min()) >= SECOND
    before = {"levels": np.array(env.levels).copy(), "spawn": np.array(env.spawn).copy()}
    _teleport(env, {0: 2.5, 6: 2.5, 2: 1.0, 8: 2.5})
    pose = np.array(env.sim.state(["go2"])["go2.pos"]).copy()
    mask = np.arange(W) < 8
    env.reset(options={"reset_mask": mask})
    out = {"before": before, "after": np.array(env.levels).copy(), "pose": pose, "mask": mask,
           "pose_after": np.array(env.sim.state(["go2"])["go2.pos"]).copy(), "tiles": _tiles(env),
           "base_after_reset": _base(env)}
    env.commands.set((0.0, 0.0, 0.0))
    _stand(env)
    out["stood"] = _base(env), np.array(env.state["base.height"]), np.array(env.state["feet.force"]).sum(1)
    return out


@pytest.mark.compile_heavy
def test_a_world_that_walked_over_half_a_tile_goes_up_one_level(sweep) -> None:
    """World 0 (level 0), moved 2.5 m (> 2 m) from its spawn, goes to level 1."""
    assert sweep["before"]["levels"][0] == 0 and sweep["after"][0] == 1


@pytest.mark.compile_heavy
def test_a_world_that_stayed_put_goes_down_one_level(sweep) -> None:
    """Worlds 3 and 4 (level 1) and 7 (level 2) stood still for 1 s at 1 m/s and drop one level; world 1 stays at 0."""
    b, a = sweep["before"]["levels"], sweep["after"]
    assert b[3] == b[4] == 1 and b[7] == 2 and b[1] == 0
    assert a[3] == 0 and a[4] == 0 and a[7] == 1 and a[1] == 0


@pytest.mark.compile_heavy
def test_a_world_in_between_keeps_its_level(sweep) -> None:
    """World 2 (level 0) walked 1.0 m of the 1 m asked (and under 2 m): its level is unchanged."""
    assert sweep["after"][2] == sweep["before"]["levels"][2] == 0


@pytest.mark.compile_heavy
def test_an_unmasked_world_keeps_level_and_pose_exactly(sweep) -> None:
    """World 8, moved 2.5 m but not in the reset mask, keeps its level and its pose bit for bit."""
    assert sweep["after"][8] == sweep["before"]["levels"][8]
    np.testing.assert_array_equal(sweep["pose_after"][8], sweep["pose"][8])


@pytest.mark.compile_heavy
def test_clearing_the_top_level_sends_the_world_to_a_random_level(sweep) -> None:
    """World 6 (top level 2) moved far: it lands on a level in [0, 2]."""
    assert sweep["before"]["levels"][6] == LEVELS - 1
    assert 0 <= sweep["after"][6] <= LEVELS - 1


@pytest.mark.compile_heavy
def test_reset_worlds_start_on_the_platform_of_their_new_tile(sweep) -> None:
    """Right after the reset each masked robot is within 0.05 m (x, y) of its new tile's centre."""
    m = sweep["mask"]
    assert np.abs(sweep["base_after_reset"][m, 0:2] - sweep["tiles"][m]).max() < 0.05


@pytest.mark.compile_heavy
def test_reset_worlds_stand_again_after_a_second(env, sweep) -> None:
    """A second after the reset the 8 masked robots are 0.264 m high on their tiles, feet carrying the weight."""
    m = sweep["mask"]
    base, height, force = sweep["stood"]
    assert np.abs(base[m, 0:2] - sweep["tiles"][m]).max() < 0.1
    assert np.abs(height[m] - 0.264).max() < 0.01
    np.testing.assert_allclose(force[m], _weight(env)[m], rtol=0.03, atol=0)


@pytest.mark.compile_heavy
def test_levels_never_change_without_the_curriculum(env) -> None:
    """With env.curriculum False a reset after a 2.5 m move and after standing still leaves every level as dealt."""
    _restart(env, curriculum=False)
    env.commands.set((1.0, 0.0, 0.0))
    _stand(env)
    _teleport(env, {0: 2.5, 6: 2.5})
    env.reset(options={"reset_mask": np.ones(W, bool)})
    np.testing.assert_array_equal(np.array(env.levels), env.initial_levels)


# --- E. off the field -----------------------------------------------------------------------------------------------

@pytest.mark.compile_heavy
def test_leaving_the_field_ends_only_that_episode_and_puts_the_robot_back(env) -> None:
    """A robot moved to x = 7.0 m (field half extent 6 m) terminates alone and restarts within 0.1 m of its tile."""
    _restart(env)
    _stand(env, 3)
    pos = _base(env).copy()
    pos[4] = (7.0, pos[4, 1], 0.3)
    env.sim.set_pose("go2", pos=pos.astype(np.float32))
    _, _, terminated, _, _ = env.step(stand_action(env))
    assert np.array(terminated).tolist() == [i == 4 for i in range(W)]
    assert np.abs(_base(env)[4, 0:2] - _tiles(env)[4]).max() < 0.1
    assert int(np.array(env.state["steps"])[4]) == 0


# --- F. the flat environment ----------------------------------------------------------------------------------------

@pytest.mark.compile_heavy
def test_flat_env_heights_are_z_and_z_minus_the_foot_radius() -> None:
    """Without terrain robot_rows.ground is None, base.height is the base z and feet.height the foot z minus the
    sole radius, exactly."""
    flat = go2_velocity(num_envs=2, device="cpu", backend="numpy", config=Go2VelocityConfig(pushes=None))
    try:
        flat.reset(seed=0)
        _stand(flat, 5)
        rows = flat.robot_rows
        assert rows.ground is None
        s = flat.state
        np.testing.assert_array_equal(np.array(s["base.height"]), flat.sim.engine.state.body_q.numpy()[rows.base][:, 2])
        feet_z = np.array(s["feet.pos"])[..., 2]
        np.testing.assert_array_equal(np.array(s["feet.height"]), feet_z - np.array(rows.foot_radius)[None, :])
    finally:
        flat.close()


def test_border_is_flat_ground_around_the_tiles() -> None:
    """A 2 m border adds 20 cells of height 0 on every side and widens the field by 2 m; tile centres stay."""
    plain, wide = Terrain(TerrainSpec(border=0.0, levels=3, tile=4.0)), Terrain(TerrainSpec(border=2.0, levels=3,
                                                                                            tile=4.0))
    assert wide.heights.shape == (plain.heights.shape[0] + 40, plain.heights.shape[1] + 40)
    assert wide.half == (plain.half[0] + 2.0, plain.half[1] + 2.0)
    assert np.array_equal(wide.heights[20:-20, 20:-20], plain.heights)
    assert wide.heights[:20].max() == 0.0 and wide.heights[:, -20:].max() == 0.0
    assert wide.tile_center(2, 1) == plain.tile_center(2, 1)
    assert bool(wide.inside(np.array([[7.0, 0.0]]))[0]) and not bool(plain.inside(np.array([[7.0, 0.0]]))[0])
