# SPDX-License-Identifier: Apache-2.0
"""Adhesion (engine item 10, v1b): sticky or electroadhesive contact, a pull over the sample points of a pad face.

It follows the MuJoCo ``<adhesion>`` actuator rule as an Alloy force. Newton's MJCF importer drops that actuator,
and NATIVECCD zeroes the geom margins it acts within, so this works with every solver instead.

    points: ``rings`` concentric rings (radii R·i/rings) of ``ring_points`` points on the pad face (the plane through
            ``center`` normal to ``axis``, in the pad body frame)
    active: a point is active on a target body when its signed distance d to one of the body's shapes has
            -``depth`` <= d <= ``gap`` and the shape normal is within ``max_angle_deg`` of the axis back at the pad
            (the suction lip test); the target with the most active points wins
    force:  n active points of N, u = max(ctrl, 0):
              spread "total" (MuJoCo):  F = u·gain, split equally over the n active points (none: 0)
              spread "area":            F = u·gain·n/N (each active point carries u·gain/N: a pad whose pull scales
                                        with the contact area, as electroadhesion and gecko pads do)
            each active point pulls the target toward the pad along the axis, the reaction on the pad body; equal
            parallel pulls act at the active points' centroid, so a partial contact patch gives a moment

``ctrl`` (per world) is the input: ``sim.plugins["adhesion"].set_input("ctrl", 1.0)``. States: ``active`` (n/N),
``target`` (index into ``targets``, -1: none) and ``force`` (the total pull [N]). The force depends on positions
only, so a closed attach-detach cycle at fixed ctrl adds no energy. The explicit step can only remove a little energy at
each crossing of ``gap``, at most F·|v|·dt (law ``energy_cycle``).

v1 limits: the pull is along the pad axis, not the local surface normal; points are equal-weight; targets are
primitives or meshes, as for suction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import warp as wp

from alloy._vendor import newton

from alloy.plugins.base import ForcePlugin, Forces
from alloy.plugins.funcs import add_wrench, body_point, body_vector, com_world, wrench_at
from alloy.plugins.harness.scene import Law, LawResult, TestScene, body_xform
from alloy.plugins.library.suction import point_sealed, target_shapes

SPREADS = ("total", "area")


@wp.kernel
def _adhesion_apply(body_q: wp.array[wp.transform], body_com: wp.array[wp.vec3], pad: wp.array[wp.int32],
                    pts: wp.array[wp.vec3], axis: wp.vec3, gap: float, depth: float, cos_max: float, gain: float,
                    area_mode: wp.int32, n_tgt: wp.int32, tgt_body: wp.array2d[wp.int32], ids: wp.array2d[wp.int32],
                    tgt: wp.array[wp.int32], shape_body: wp.array[wp.int32], shape_type: wp.array[wp.int32],
                    shape_scale: wp.array[wp.vec3], shape_xf: wp.array[wp.transform], shape_mesh: wp.array[wp.uint64],
                    ctrl: wp.array[wp.float32],
                    active: wp.array[wp.float32], target: wp.array[wp.int32], force: wp.array[wp.float32],
                    body_f: wp.array[wp.spatial_vector]):
    w = wp.tid()
    pb = pad[w]
    ax = wp.normalize(body_vector(body_q, pb, axis))
    n_pts = pts.shape[0]
    best, best_t = int(0), int(-1)
    best_sum = wp.vec3(0.0)
    for t in range(n_tgt):
        count = int(0)
        acc = wp.vec3(0.0)
        for k in range(n_pts):
            p = body_point(body_q, pb, pts[k])
            if point_sealed(p, -ax, w, t, body_q, ids, tgt, shape_body, shape_type, shape_scale, shape_xf, shape_mesh,
                            gap, depth, cos_max) == 1:
                count += 1
                acc = acc + p
        if count > best:
            best, best_t, best_sum = count, t, acc
    f = float(0.0)
    if best > 0:
        f = wp.max(ctrl[w], 0.0) * gain
        if area_mode == 1:
            f = f * float(best) / float(n_pts)
        c = best_sum / float(best)
        tb = tgt_body[w, best_t]
        add_wrench(body_f, tb, wrench_at(-f * ax, c, com_world(body_q, body_com, tb)))
        add_wrench(body_f, pb, wrench_at(f * ax, c, com_world(body_q, body_com, pb)))
    active[w] = float(best) / float(n_pts)
    target[w] = best_t
    force[w] = f


def pad_points(radius: float, rings: int, ring_points: int, center: tuple[float, float, float],
               axis: tuple[float, float, float]) -> np.ndarray:
    """``[rings·ring_points, 3]`` sample points in the pad body frame (ring i at radius R·i/rings)."""
    ax = np.asarray(axis, np.float64) / np.linalg.norm(axis)
    ref = np.array([0.0, 1.0, 0.0]) if abs(ax[0]) > 0.9 else np.array([1.0, 0.0, 0.0])
    u1 = np.cross(ax, ref)
    u1 /= np.linalg.norm(u1)
    u2 = np.cross(ax, u1)
    phi = 2.0 * np.pi * np.arange(ring_points) / ring_points
    ring = np.cos(phi)[:, None] * u1 + np.sin(phi)[:, None] * u2
    return np.concatenate([np.asarray(center, np.float64) + radius * (i / rings) * ring for i in range(1, rings + 1)])


class Adhesion(ForcePlugin):
    """Sticky or electroadhesive pad: the pull ``ctrl·gain`` over the pad points within ``gap`` of a target, on the
    target toward the pad and back on the pad (MuJoCo's ``<adhesion>`` rule; ``spread="area"`` scales it by the
    active share).

    Example::

        adh = Adhesion(body="gripper_pad", center=(0, 0, -0.005), gain=20.0, targets=("sheet",))
        sim = alloy.plugins.simulation(built, [adh]); adh.set_input("ctrl", 1.0)
    """

    name = "adhesion"
    summary = "sticky / electroadhesive pad: ctrl·gain pull over the pad points within a gap of a target (MuJoCo rule)"
    internal = True

    @dataclass
    class Config:
        body: str = "adhesion_pad"                            # body label of the pad (takes the reaction)
        center: tuple[float, float, float] = (0.0, 0.0, 0.0)  # m, pad face centre in the body frame
        axis: tuple[float, float, float] = (0.0, 0.0, -1.0)   # out of the face, in the body frame
        radius: float = 0.02                                  # m, outer ring radius
        rings: int = 1
        ring_points: int = 16
        gap: float = 1.0e-3                                   # m above the surface (MuJoCo's adhesion margin)
        depth: float = 5.0e-3                                 # m into the surface (soft contacts sink)
        max_angle_deg: float = 30.0                           # surface normal vs the axis back at the pad
        targets: tuple[str, ...] = ("box",)                   # body labels the pad may stick to
        gain: float = 10.0                                    # N at ctrl 1 (the whole pad)
        spread: str = "total"                                 # "total" (MuJoCo) or "area"

        def __post_init__(self) -> None:
            if self.rings < 1 or self.ring_points < 3:
                raise ValueError(f"rings >= 1 and ring_points >= 3, got {self.rings} and {self.ring_points}")
            if min(self.radius, self.gap) <= 0 or min(self.depth, self.gain) < 0:
                raise ValueError("radius and gap must be > 0; depth and gain >= 0")
            if self.spread not in SPREADS:
                raise ValueError(f"spread must be one of {SPREADS}, got {self.spread!r}")
            if not self.targets:
                raise ValueError("targets: at least one body label")

    @property
    def point_count(self) -> int:
        return self.config.rings * self.config.ring_points

    def pull(self, ctrl: float, share: float = 1.0) -> float:
        """The total pull [N] at ``ctrl`` with ``share`` of the points active (> 0)."""
        f = max(ctrl, 0.0) * self.config.gain
        return f * share if self.config.spread == "area" else f

    def setup(self, ctx: Any) -> None:
        c, W = self.config, ctx.world_count
        self.pad = ctx.bodies(c.body)
        self.tgt_body, self.ids, self.tgt = target_shapes(ctx, c.targets, self.label)
        if np.any(self.tgt_body.numpy() == self.pad.numpy()[:, None]):
            raise ValueError(f"{self.label}: the pad body {c.body!r} cannot be one of its own targets")
        self.pts = ctx.array(pad_points(c.radius, c.rings, c.ring_points, c.center, c.axis).astype(np.float32),
                             wp.vec3)
        self.active = self.add_state("active", ctx.zeros(W))
        self.target = self.add_state("target", ctx.full(W, -1, wp.int32))
        self.force = self.add_state("force", ctx.zeros(W))
        self.ctrl = self.add_input("ctrl", ctx.zeros(W))

    def apply(self, state: newton.State, forces: Forces, dt: float) -> None:
        c, m = self.config, self.ctx.model
        wp.launch(_adhesion_apply, dim=self.ctx.world_count,
                  inputs=[state.body_q, m.body_com, self.pad, self.pts, wp.vec3(*c.axis), float(c.gap), float(c.depth),
                          float(math.cos(math.radians(c.max_angle_deg))), float(c.gain), int(c.spread == "area"),
                          len(c.targets), self.tgt_body, self.ids, self.tgt, m.shape_body, m.shape_type, m.shape_scale,
                          m.shape_transform, m.shape_source_ptr, self.ctrl],
                  outputs=[self.active, self.target, self.force, forces.body_f], device=self.ctx.device)

    def test_scene(self) -> TestScene:
        return hold_scene(self, 0.9 * self.pull(1.0) / G)

    def laws(self) -> list[Law]:
        return adhesion_laws(self)


# ---- law scenes and checks (A-ADH-1, the moment of a partial patch, the energy cycle, the null case) ----------------

G = 9.81
BOX = (0.04, 0.04, 0.02)               # box half extents [m]
PAD_Z = 0.30                           # pad face height in the hanging scenes
PAD_H = 0.005                          # pad half thickness
HOLD_TOL, DROP_MIN = 1.0e-3, 5.0e-3    # m: held = moved less than this; let go = moved more than this
CONTACT_KE = 1.0e5
CONTACT_KD = 2.0 * CONTACT_KE ** 0.5   # critical contact damping (Newton's default kd chatters under a steady load)
NARROW = 0.75                          # narrow box half-width / R: the 6 points at |cos φ| >= 0.924 hang off
NARROW_SHARE = 10.0 / 16.0
# energy cycle: a 0.5 kg box (no collisions) on a vertical spring at 2 Hz, rest 8 mm under the face, amplitude 6 mm;
# gap 3 mm, pull 0.2 N: it dips 1.7 mm into the zone and turns 1.3 mm short of the face
E_M, E_HZ, E_REST, E_AMP, E_GAP, E_CTRL = 0.5, 2.0, 8.0e-3, 6.0e-3, 3.0e-3, 0.02
SPREAD_X = BOX[0] + 0.3 * 0.02         # moment law: box centre x, its -x edge at 0.3·R (7 of 16 points over it)


def _pad(b: newton.ModelBuilder, plug: Any, z_face: float) -> None:
    """A pad fixed to the world, its face centre (``config.center``) at (0, 0, ``z_face``), facing -z."""
    c = plug.config
    origin = body_xform(-c.center[0], -c.center[1], z_face - c.center[2])
    pad = b.add_link(xform=origin, label=c.body)
    b.add_shape_box(pad, xform=body_xform(c.center[0], c.center[1], c.center[2] + PAD_H), hx=c.radius, hy=c.radius,
                    hz=PAD_H, cfg=newton.ModelBuilder.ShapeConfig(density=500.0, ke=CONTACT_KE, kd=CONTACT_KD))
    b.add_articulation([b.add_joint_fixed(-1, pad, parent_xform=origin, label=f"{c.body}_base")], label=c.body)


def _box(b: newton.ModelBuilder, mass: float, center: tuple[float, float, float], half=BOX) -> int:
    body = b.add_body(xform=body_xform(*center), label="box")
    b.add_shape_box(body, hx=half[0], hy=half[1], hz=half[2],
                    cfg=newton.ModelBuilder.ShapeConfig(density=mass / (8.0 * half[0] * half[1] * half[2]), ke=CONTACT_KE,
                                                        kd=CONTACT_KD))
    return body


def _ctrl(label: str, values: Any):
    def init(sim: Any) -> None:
        if label in sim.plugins.plugins:                 # the speed baseline runs without the plug-in
            sim.plugins[label].set_input("ctrl", values)
    return init


def hold_scene(plug: Any, mass: float, half=BOX, ctrl: float = 1.0, periods: int = 25) -> TestScene:
    """A box of ``mass`` hanging centred under the fixed pad, its top on the face."""
    def build(b: newton.ModelBuilder) -> None:
        _pad(b, plug, PAD_Z)
        _box(b, mass, (0.0, 0.0, PAD_Z - half[2]), half=half)

    return TestScene(build=build, init=_ctrl(plug.label, ctrl), periods=periods, ground=False)


def rest_scene(plug: Any, gap: float, x: float = 0.0, ctrl: float = 1.0, periods: int = 10) -> TestScene:
    """A heavy box (20 kg) on the ground at x, the pad face ``gap`` above its top."""
    def build(b: newton.ModelBuilder) -> None:
        _pad(b, plug, 2.0 * BOX[2] + gap)
        _box(b, 20.0, (x, 0.0, BOX[2]))

    return TestScene(build=build, init=_ctrl(plug.label, ctrl), periods=periods)


def energy_scene(plug: Any, periods: int = 150) -> TestScene:
    """No gravity; the box (no collisions) on a vertical prismatic spring under the pad, released 6 mm below its rest
    point: it swings into the adhesion zone and out every cycle. World 1 has ctrl 0 (the integrator's own band)."""
    k = E_M * (2.0 * math.pi * E_HZ) ** 2
    z_rest = PAD_Z - E_REST - BOX[2]

    def build(b: newton.ModelBuilder) -> None:
        _pad(b, plug, PAD_Z)
        box = b.add_link(xform=body_xform(0.0, 0.0, z_rest - E_AMP), label="box")
        cfg = newton.ModelBuilder.ShapeConfig(density=E_M / (8.0 * BOX[0] * BOX[1] * BOX[2]), has_shape_collision=False)
        b.add_shape_box(box, hx=BOX[0], hy=BOX[1], hz=BOX[2], cfg=cfg)
        j = b.add_joint_prismatic(-1, box, parent_xform=body_xform(0.0, 0.0, z_rest), axis=(0.0, 0.0, 1.0),
                                  target_pos=0.0, target_ke=k, target_kd=0.0, label="box_spring")
        b.joint_q[-1] = -E_AMP
        b.add_articulation([j], label="box_spring")

    return TestScene(build=build, init=_ctrl(plug.label, [E_CTRL, 0.0, E_CTRL]), periods=periods, ground=False,
                     gravity=(0.0, 0.0, 0.0))


def _drop(run: Any) -> float:
    z = run.trace.body_q[:, int(run.bodies("box")[0]), 2].astype(np.float64)
    return float(np.max(np.abs(z[0] - z)))


def _held(want: bool, what: str, share: float | None = None):
    def check(run: Any) -> LawResult:
        d = _drop(run)
        s = float(run.trace.plugin(run.plugin.label, "active")[1, 0])
        share_ok = share is None or abs(s - share) < 1e-6
        tail = f"; active {s:.4f}" + ("" if share is None else f" (geometry {share:.4f})")
        if want:
            return LawResult(d < HOLD_TOL and share_ok, d, HOLD_TOL,
                             f"{what}: box moved {d * 1e3:.3g} mm (held: < {HOLD_TOL * 1e3:g} mm){tail}",
                             "pull = ctrl·gain (total) or ctrl·gain·n/N (area) on the box toward the pad")
        return LawResult(d > DROP_MIN and share_ok, d, DROP_MIN,
                         f"{what}: box moved {d * 1e3:.3g} mm (let go: > {DROP_MIN * 1e3:g} mm){tail}",
                         "no hold above the pull: check the spread rule")
    return check


def _check_moment(run: Any) -> LawResult:
    """Probe the pull on the offset box: u·gain at the centroid of the points over its top (numpy footprint test)."""
    plug, c = run.plugin, run.plugin.config
    q0 = run.trace.body_q[0].astype(np.float64)
    b, p = int(run.bodies("box")[0]), int(run.bodies(c.body)[0])
    from alloy.plugins.npgeom import transform_point  # noqa: PLC0415

    pts = np.array([transform_point(q0[p], x) for x in pad_points(c.radius, c.rings, c.ring_points, c.center, c.axis)])
    over = np.all(np.abs(pts[:, :2] - q0[b, :2]) <= np.array(BOX[:2]), axis=1)
    ctrl = float(run.host[plug.label].inputs()["ctrl"].numpy()[0])
    f_want = plug.pull(ctrl, over.mean())
    com = transform_point(q0[b], run.model.body_com.numpy()[b].astype(np.float64))
    want = np.concatenate([[0.0, 0.0, f_want], np.cross(pts[over].mean(axis=0) - com, [0.0, 0.0, f_want])])
    got = run.probe(run.trace.body_q[0])[b]
    err = float(np.max(np.abs(got - want)) / f_want)
    return LawResult.within(err, 1e-4, f"{int(over.sum())} of {len(pts)} points over the box: force {got[2]:.5g} N "
                            f"(want {f_want:.5g}), moment y {got[4]:.5g} N·m (want {want[4]:.5g}); worst {err:.2e} "
                            "relative", "equal pulls at the active points: the resultant acts at their centroid")


def _energy(run: Any, w: int) -> np.ndarray:
    c, tr = run.plugin.config, run.trace
    b = int(run.bodies("box")[w])
    z, v = tr.body_q[:, b, 2].astype(np.float64), tr.body_qd[:, b, 2].astype(np.float64)
    k = E_M * (2.0 * math.pi * E_HZ) ** 2
    d = PAD_Z - (z + BOX[2])                                # face to box top
    f = run.plugin.pull(float(run.host[run.plugin.label].inputs()["ctrl"].numpy()[w]))
    u = np.where((d <= c.gap) & (d >= -c.depth), -f * (c.gap - d), 0.0)
    m = float(run.model.body_mass.numpy()[b])
    return 0.5 * m * v * v + 0.5 * k * (z - z[0] - E_AMP) ** 2 + u


def _check_energy(run: Any) -> LawResult:
    """Max energy over the last cycle <= max over the first + the ctrl-0 world's own band; the loss is at most
    F·v·dt per gap crossing."""
    c = run.plugin.config
    t = np.asarray(run.trace.t, np.float64)
    e, e0 = _energy(run, 0), _energy(run, 1)
    band = float(e0.max() - e0.min())
    cyc = 1.0 / E_HZ
    first, last = e[t <= cyc].max(), e[t >= t[-1] - cyc].max()
    d = PAD_Z - (run.trace.body_q[:, int(run.bodies("box")[0]), 2].astype(np.float64) + BOX[2])
    inside = d <= c.gap
    crossings = int(np.count_nonzero(inside[1:] != inside[:-1]))
    f = run.plugin.pull(E_CTRL)
    k = E_M * (2.0 * math.pi * E_HZ) ** 2
    v_in = math.sqrt(k / E_M * (E_AMP ** 2 - (E_REST - c.gap) ** 2))
    dt = run.scene.control_dt / 2 ** int(run.sim.clock.j)
    loss_max = crossings * f * v_in * dt + band
    ok = last - first <= band and first - last <= loss_max and crossings >= 8 and float(d.min()) > 0.0
    return LawResult(ok, (last - first) / first, band / first,
                     f"{crossings} gap crossings over {t[-1]:.2f} s, closest {d.min() * 1e3:.2f} mm: max energy "
                     f"{first * 1e3:.5g} mJ (first cycle) -> {last * 1e3:.5g} mJ (last); ctrl-0 band "
                     f"{band * 1e3:.2g} mJ; allowed loss {loss_max * 1e3:.2g} mJ (F·v·dt per crossing)",
                     "the pull must depend on positions only (no lag, no velocity term)")


def _check_null(run: Any) -> LawResult:
    f, s = (run.trace.plugin(run.plugin.label, n).astype(np.float64) for n in ("force", "active"))
    worst = float(np.max(np.abs(f)) + np.max(np.abs(s)))
    return LawResult(worst == 0.0, worst, 0.0, f"pad 5 mm above the box: |force| + |active| max {worst:.3g}",
                     "a point is active only within gap of a target surface")


def adhesion_laws(plug: Any) -> list[Law]:
    """A-ADH-1 (pull-off = ctrl·gain over the active points, 2 %), its area variant, the moment of a partial patch,
    the energy cycle and the null case."""
    f = plug.pull(1.0)
    r = plug.config.radius
    narrow = (NARROW * r, BOX[1], BOX[2])
    area = {"spread": "area"}
    f_area = f * NARROW_SHARE
    return [
        Law("A-ADH-1_holds_below", _held(True, "0.98·ctrl·gain", 1.0), hold_scene(plug, 0.98 * f / G),
            "a pad on a wide box holds m·g = 0.98·ctrl·gain"),
        Law("A-ADH-1_lets_go_above", _held(False, "1.02·ctrl·gain", 1.0), hold_scene(plug, 1.02 * f / G),
            "and lets go at 1.02"),
        Law("A-ADH-1_partial_holds_below", _held(True, "10 of 16 points, 0.98·ctrl·gain", NARROW_SHARE),
            hold_scene(plug, 0.98 * f / G, half=narrow), "MuJoCo's rule: the whole ctrl·gain over the active points"),
        Law("A-ADH-1_partial_lets_go_above", _held(False, "10 of 16 points, 1.02·ctrl·gain", NARROW_SHARE),
            hold_scene(plug, 1.02 * f / G, half=narrow), "and no more"),
        Law("area_holds_below", _held(True, "area: 0.98·ctrl·gain·10/16", NARROW_SHARE),
            hold_scene(plug, 0.98 * f_area / G, half=narrow), "spread 'area': the pull scales with the active share",
            area),
        Law("area_lets_go_above", _held(False, "area: 1.02·ctrl·gain·10/16", NARROW_SHARE),
            hold_scene(plug, 1.02 * f_area / G, half=narrow), "and lets go at 1.02", area),
        Law("A-ADH-1_partial_moment", _check_moment, rest_scene(plug, 5.0e-4, x=SPREAD_X, ctrl=0.5),
            "box under part of the pad, ctrl 0.5: force 0.5·gain at the active points' centroid (1e-4)"),
        Law("energy_cycle", _check_energy, energy_scene(plug),
            "a box on a spring swings in and out of the zone: no energy gain over 6 cycles", {"gap": E_GAP}),
        Law("null_out_of_reach", _check_null, rest_scene(plug, 5.0e-3), "pad 5 mm above the box: no force"),
    ]


__all__ = ["Adhesion", "adhesion_laws", "energy_scene", "hold_scene", "pad_points", "rest_scene"]
