# SPDX-License-Identifier: Apache-2.0
"""OpenTac board reading (alloy/plugins/tactile/opentac): manifest loading, parity with OpenTac's fixtures, the bump
split and decoder round trip, the readout, and the Warp kernels against the NumPy reference (CPU, no solver).

The fixtures in test_tactile_opentac_data/ are copied unchanged from github.com/opentac/opentac tests/sim/fixtures at
commit e302861 (MIT, Copyright (c) 2026 The OpenTac authors). The laws on solver scenes run in the harness quick
check (alloy/plugins/tactile/opentac/laws.py)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import warp as wp

from alloy.plugins.tactile import Tactile
from alloy.plugins.tactile.layout import build_layout
from alloy.plugins.tactile.opentac import BOARDS, ManifestError, decode, load_board, profile, resolve
from alloy.plugins.tactile.opentac import foam, kernels as K, model as M, readout as R
from alloy.plugins.tactile.opentac.reading import _count_args

DATA = Path(__file__).parent / "test_tactile_opentac_data"
FOAM = json.loads((DATA / "foam_reference.json").read_text())
PITCH_UM = {"flat_square.json": 2500, "flat_disc.json": 2500, "grid_4mm.json": 4000, "grid_2mm.json": 2000}


def _raw_xy(skin: str) -> np.ndarray:
    return np.asarray([p[:2] for p in json.loads((DATA / skin).read_text())["pixels"]], float)


# -- manifests ---------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(BOARDS))
def test_presets_load_with_their_sources(name):
    b = resolve(name)
    assert BOARDS[name].source and b.bumps and b.n_bumps > 0
    assert b.group_sizes().sum() == (b.taxel_bump >= 0).sum()
    np.testing.assert_allclose((b.xy.min(0) + b.xy.max(0)) / 2, 0.0, atol=1e-12)    # centred pad frame
    assert build_layout(Tactile.Config(opentac=name)).reach == pytest.approx(b.reach)


def test_preset_numbers_match_their_manifests():
    got = {n: (resolve(n).n_taxels, resolve(n).n_bumps, round(resolve(n).pitch[0] * 1e3, 3)) for n in BOARDS}
    assert got == {"opentac_finger_8x6_60pin": (45, 12, 3.0), "opentac_pinky_tip_30pin": (32, 8, 2.0),
                   "opentac_square_9x9_30pin": (81, 25, 2.0), "opentac_square_9x9_60pin": (81, 25, 4.2)}


@pytest.mark.parametrize("skin", sorted(PITCH_UM))
def test_pitch_matches_their_derivation(skin):
    """transfer_reference.json records their pitch derivation for each fixture skin."""
    assert json.loads((DATA / "transfer_reference.json").read_text())["pitch_um"][skin] == [PITCH_UM[skin]] * 2
    assert round(load_board(DATA / skin, bumps=False).pitch[0] * 1e6) == PITCH_UM[skin]


def test_curved_and_shearless_manifests_are_refused():
    m = json.loads((DATA / "flat_square.json").read_text())
    with pytest.raises(ManifestError, match="curved surface"):
        load_board({**m, "sim3d": {"taxels3d": [[0, 0, 0]]}})
    with pytest.raises(ManifestError, match="no usable 'shear'"):
        load_board(DATA / "grid_4mm.json")
    assert not load_board(DATA / "grid_4mm.json", bumps=False).bumps


@pytest.mark.parametrize("change, message", [
    ({"units": "in"}, "mm"), ({"shear": {"bumps": [[0, 0]], "b": None}}, "no 'b'"),
    ({"bump_params": {"taxel_pitch_mm": [1, 1]}}, "taxel_pitch_mm"),
    ({"bump_params": {"bump_base_frac": 2.0}}, "overlap"),
])
def test_malformed_manifests_are_refused(change, message):
    m = {**json.loads((DATA / "flat_square.json").read_text()), **change}
    with pytest.raises(ManifestError, match=message):
        load_board(m)


def test_config_refuses_a_second_layout_and_unknown_names():
    with pytest.raises(ValueError, match="leave preset"):
        build_layout(Tactile.Config(opentac="square_9x9_60pin", preset="reskin"))
    with pytest.raises(ValueError, match="no OpenTac board"):
        resolve("nope")
    with pytest.raises(ValueError, match="no OpenTac readout profile"):
        profile("v9")
    assert profile("yunjia").name == "v3" and profile("LINQSTAT").name == "v2"


# -- foam parity ---------------------------------------------------------------------------------------------------

def _args(case: dict) -> tuple:
    board = load_board(DATA / case["skin"], bumps=False)
    return (_raw_xy(case["skin"]), board.pitch[0] * 1e3, case["geom"], case["centre_mm"], case["total_n"],
            case["sigma_mm"], case["angle_deg"], case["k_n_per_mm3"]), board


@pytest.mark.parametrize("case", FOAM["cases"], ids=lambda c: f"{c['skin']}-{c['geom']['kind']}-{c['total_n']}N")
def test_foam_port_reproduces_their_fixture(case):
    args, _ = _args(case)
    np.testing.assert_allclose(foam.taxel_forces(*args), case["forces_n"], rtol=1e-12, atol=1e-15)


@pytest.mark.parametrize("case", FOAM["cases"], ids=lambda c: f"{c['skin']}-{c['geom']['kind']}-{c['total_n']}N")
def test_contact_route_matches_their_fixture_within_one_percent_of_peak(case):
    """Their pressure as point loads through Alloy's route (exact Gaussian and squares, not their 0.5 mm grid)."""
    args, board = _args(case)
    pts, f = foam.winkler_points(*args)
    got = M.foam_loads(board, (pts - board.centre_mm) * 1e-3, f, "opentac", case["sigma_mm"])
    want = np.asarray(case["forces_n"])
    assert np.abs(got - want).max() <= 0.01 * want.max()


def test_their_foam_squares_overcount_off_the_odd_half_millimetre_pitches():
    """Their taxel squares are 2 * round(p / 1 mm) + 1 half-millimetres wide: 2.5 mm at 2.5 mm pitch, 4.5 mm at 4 mm,
    2.5 mm at 2 mm; a dense board then reads (width / pitch)^2 of the force. Their fixture shows it."""
    assert [M.foam_box_factor(p) for p in (2.5e-3, 4e-3, 2e-3)] == pytest.approx([1.0, 1.265625, 1.5625])
    case = FOAM["cases"][6]                                       # grid_2mm, 1 N, sphere well inside the board
    assert sum(case["forces_n"]) / case["total_n"] == pytest.approx(M.foam_box_factor(2e-3), rel=0.03)


# -- bump split, decoder, readout ----------------------------------------------------------------------------------

def _full_block(board):
    return int(np.flatnonzero(board.group_sizes() == 4)[0])


def test_split_conserves_each_bump_load_and_decodes_pure_shear_exactly():
    board = resolve("square_9x9_60pin")
    rng = np.random.default_rng(0)
    fz = rng.uniform(0.1, 2.0, board.n_bumps)
    shear = rng.uniform(-0.3, 0.3, (board.n_bumps, 2)) * fz[:, None]
    wrench = np.column_stack([fz, shear, np.zeros((board.n_bumps, 2))])
    load = M.split_wrench(board, fz, M.base_moments(wrench, board.bump_height))
    np.testing.assert_allclose(np.bincount(board.taxel_bump, load, board.n_bumps), fz, rtol=1e-12)
    got = M.decode_shear(board, load)
    ok = ~M.degenerate_axes(board)
    np.testing.assert_allclose(got[ok], shear[ok], atol=1e-12)


def test_decoded_shear_saturates_past_pitch_over_height():
    """A full 2x2 block clips once F_t / F_n passes taxel pitch / bump height (4.2 / 2.0 here)."""
    board, ratio = resolve("square_9x9_60pin"), 4.2 / 2.0
    k = _full_block(board)
    for r, exact in ((0.95 * ratio, True), (1.2 * ratio, False)):
        wrench = np.zeros((board.n_bumps, 5))
        wrench[k, :2] = (1.0, r)
        got = M.decode_shear(board, M.split_wrench(board, wrench[:, 0], M.base_moments(wrench, board.bump_height)))
        assert bool(got[k, 0] == pytest.approx(r, rel=1e-9)) is exact
        assert got[k, 0] <= r + 1e-9


def test_off_centre_load_decodes_as_shear():
    board = resolve("square_9x9_60pin")
    k = _full_block(board)
    w, _ = M.bump_wrench(board, board.bump_xy[k:k + 1] + (0.0005, 0.0), np.array([[1.0, 0.0, 0.0]]))
    got = M.decode_shear(board, M.split_wrench(board, w[:, 0], M.base_moments(w, board.bump_height, tip=True)))
    assert got[k] == pytest.approx((2 * 0.5 / 2.0, 0.0), abs=1e-9)          # 2 F dx / H
    flat = M.decode_shear(board, M.split_wrench(board, w[:, 0], M.base_moments(w, board.bump_height)))
    assert flat[k] == pytest.approx((0.0, 0.0), abs=1e-12)                   # default: no tip term


@pytest.mark.parametrize("name", ["v2", "v3"])
def test_counts_rise_clip_and_invert(name):
    p, pitch = profile(name), (0.0042, 0.0042)
    f = np.geomspace(1e-3, 50.0, 400)
    c = R.counts(f, pitch, p, quantize=False)
    assert np.all(np.diff(c) >= 0) and c[-1] == p.adc_max and c[0] < p.adc_max
    keep = (c < p.adc_max) & (c > 3 * p.noise_counts)
    np.testing.assert_allclose(R.forces(c, pitch, p)[keep], f[keep], rtol=1e-9)
    assert R.saturation_force_n(pitch, p) == pytest.approx(f[np.argmax(c >= p.adc_max)], rel=0.05)


def test_round_trip_through_counts_and_the_decoder():
    """Wrench to v3 counts (whole counts) to decoded shear: within 1 % below the clip, on every full block."""
    board = resolve("finger_8x6_60pin")
    full = np.flatnonzero(board.group_sizes() == 4)
    wrench = np.zeros((board.n_bumps, 5))
    wrench[full, 0], wrench[full, 1], wrench[full, 2] = 1.0, 0.4, -0.25
    load = M.split_wrench(board, wrench[:, 0], M.base_moments(wrench, board.bump_height))
    out = decode(board, R.counts(load, board.pitch, profile("v3")), "v3")
    np.testing.assert_allclose(out["bump_shear"][full], wrench[full, 1:3], rtol=0.01)
    np.testing.assert_allclose(out["bump_normal"][full], 1.0, rtol=0.01)


# -- Warp kernels against the NumPy reference ----------------------------------------------------------------------

def _launch(board, points, forces, cells="opentac", readout="v3", tip=False):
    """bin_board -> split_bumps -> finish_board on one pad (body 0, pad frame = world), contacts on shape 0."""
    from alloy.plugins.tactile.opentac.reading import BoardReading

    dev, n, T, B = "cpu", len(points), board.n_taxels, board.n_bumps
    cfg = Tactile.Config(opentac="square_9x9_60pin", opentac_readout=readout, opentac_foam_cells=cells, opentac_tip=tip)
    reading = BoardReading(board, cfg)

    class Ctx:
        @staticmethod
        def array(a, dtype):
            return wp.array(a, dtype=dtype, device=dev)
    a = reading._board_args(Ctx, board.reach)
    f = np.asarray(forces, float)
    cf = np.zeros((n, 6), np.float32)
    cf[:, :3] = np.column_stack([f[:, 1], f[:, 2], -f[:, 0]])                  # force on the pad, world
    pts = np.column_stack([points, np.zeros(n)]).astype(np.float32)
    bacc, tacc = wp.zeros(max(B * 5, 1), wp.int64, device=dev), wp.zeros(T, wp.int64, device=dev)
    load, counts = wp.zeros(T, wp.float32, device=dev), wp.zeros(T, wp.float32, device=dev)
    wrench = wp.zeros(max(B * 5, 1), wp.float32, device=dev)
    wp.launch(K.bin_board, dim=n, device=dev, inputs=[
        wp.array([n], wp.int32, device=dev), wp.zeros(n, wp.int32, device=dev), wp.full(n, -1, wp.int32, device=dev),
        wp.array(pts, wp.vec3, device=dev), wp.array(pts, wp.vec3, device=dev),
        wp.array(cf, wp.spatial_vector, device=dev), wp.zeros(1, wp.int32, device=dev),
        wp.array([wp.transform_identity()], wp.transform, device=dev), wp.zeros(1, wp.int32, device=dev), 1,
        wp.zeros(1, wp.vec3, device=dev), wp.array([wp.quat_identity()], wp.quat, device=dev), a],
        outputs=[bacc, tacc])
    if B:
        wp.launch(K.split_bumps, dim=B, device=dev, inputs=[
            bacc, wp.array(board.member_start.astype(np.int32), wp.int32, device=dev),
            wp.array(board.members.astype(np.int32), wp.int32, device=dev),
            wp.array(board.moment_weights.astype(np.float32), wp.vec2, device=dev), B, T, float(board.bump_height),
            1.0 if tip else 0.0],
            outputs=[load, wrench])
    wp.launch(K.finish_board, dim=T, device=dev, inputs=[tacc, a.taxel_bump, T, T, wp.zeros(1, wp.int32, device=dev),
                                                         _count_args(board, cfg)], outputs=[load, counts])
    return load.numpy(), counts.numpy(), wrench.numpy()[:B * 5].reshape(B, 5)


def _contacts(board, n=40, seed=1):
    rng = np.random.default_rng(seed)
    pts = rng.uniform(-0.6, 0.6, (n, 2)) * np.ptp(board.xy, 0)
    f = np.column_stack([rng.uniform(0.0, 0.4, n), rng.uniform(-0.1, 0.1, (n, 2))])
    return pts, f


@pytest.mark.parametrize("tip", [False, True])
@pytest.mark.parametrize("name", sorted(BOARDS))
def test_kernels_match_the_reference_on_bump_boards(name, tip):
    board = resolve(name)
    pts, f = _contacts(board)
    load, counts, wrench = _launch(board, pts, f, tip=tip)
    ref = M.board_loads(board, pts, f, tip=tip)
    np.testing.assert_allclose(load, ref, rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(counts, R.counts(ref, board.pitch, profile("v3")), atol=1.0)
    w, _ = M.bump_wrench(board, pts, f)
    np.testing.assert_allclose(wrench[:, :3], w[:, :3], atol=1e-6)
    np.testing.assert_allclose(wrench[:, 3:], w[:, 3:], atol=1e-8)


@pytest.mark.parametrize("cells", ["opentac", "pitch"])
def test_kernels_match_the_reference_on_a_plain_skin(cells):
    board = load_board(DATA / "grid_4mm.json", bumps=False)
    pts, f = _contacts(board)
    load, counts, _ = _launch(board, pts, f, cells=cells, readout="v2")
    ref = M.board_loads(board, pts, f, cells)
    np.testing.assert_allclose(load, ref, rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(counts, R.counts(ref, board.pitch, profile("v2")), atol=1.0)


def test_plugin_adds_the_board_readings_and_keeps_the_newton_ones():
    """On the tactile rest scene (2 worlds): the new readings have their shapes, the newton channels are the same
    as without the board, and the board's laws are registered with the plug-in."""
    import dataclasses  # noqa: PLC0415

    from alloy.plugins.harness.run import run_scene  # noqa: PLC0415
    from alloy.plugins.tactile.laws import rest_scene  # noqa: PLC0415

    scene = dataclasses.replace(rest_scene(), worlds=2, periods=5)
    plain = run_scene(scene, [Tactile(points=tuple(map(tuple, resolve("square_9x9_60pin").xy)))], "cpu")
    run = run_scene(scene, [Tactile(opentac="square_9x9_60pin")], "cpu")
    assert run.error is None and plain.error is None, (run.error, plain.error)
    read = lambda r, k: np.asarray(r.trace.plugin(r.plugin.label, k))[-1]  # noqa: E731
    assert read(run, "counts").shape == read(run, "board_load").shape == (2, 1, 81)
    assert read(run, "bump_wrench").shape == (2, 1, 25, 5)
    for k in ("taxels", "pad_force"):
        np.testing.assert_array_equal(read(run, k), read(plain, k))
    assert read(run, "counts").max() > 0
    assert any(law.name == "opentac_round_trip" for law in Tactile().laws())
