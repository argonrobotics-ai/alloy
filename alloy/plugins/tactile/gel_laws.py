# SPDX-License-Identifier: Apache-2.0
"""GelSensor's laws (tactile v2 physics; ruling 2026-11-06), on the v1 pad scenes (``laws.py``).

- **depth_cap:** a 5 mm sphere resting on the gel. The depth field equals the analytic spherical cap at its
  penetration, read from the two bodies' poses (the engine's own indentation), within 2 % of that penetration.
- **force_weight:** the engine's normal force on the gel equals m·g, within 0.5 %.
- **markers_normal / markers_shear / markers_twist:** the marker field equals an independent numpy evaluation of the
  same model (loads from the final depth field and the engine's force and twist; the same Green's tensor), within 1 %
  of the largest displacement. The scenes are the sphere press, a box under a 0.3 g tangential load, and a box under a
  twisting moment.
- **twist_sense:** under a twisting moment, the markers circulate in its sense: Σ (q × u)_z has the sign of the
  moment on the gel.
- **null:** nothing on the gel: depth, markers and force all exactly 0.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import warp as wp

from alloy._vendor import newton

from alloy.plugins.base import ForcePlugin, Forces
from alloy.plugins.funcs import add_wrench
from alloy.plugins.harness.scene import Law, LawResult, TestScene, body_xform
from alloy.plugins.tactile import laws as L1

R_BALL = 0.005
MARKERS = {"markers": (5, 7)}
TWIST = 3.0e-4          # N·m about +z, below the box's static friction limit (~1e-3 N·m)


def _build_press(b: newton.ModelBuilder) -> None:
    L1._pad(b)
    s = b.add_body(xform=body_xform(0.0007, -0.0011, 2 * L1.PAD_H + R_BALL + 5e-4), label="ball")
    b.add_shape_sphere(s, radius=R_BALL, cfg=L1._cfg(0.05 / (4.0 / 3.0 * math.pi * R_BALL ** 3)))


def press_scene() -> TestScene:
    """A 5 mm, 50 g sphere resting on the gel pad (the plug-in's default test scene)."""
    return TestScene(build=_build_press, periods=25, ground_mu=1.0)


@wp.kernel
def _torque(body: wp.array(dtype=wp.int32), tau: float, body_f: wp.array(dtype=wp.spatial_vector)):
    w = wp.tid()
    add_wrench(body_f, body[w], wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, tau))


class TwistTorque(ForcePlugin):
    """Test companion: a constant torque about world +z on body ``box`` [N·m]."""

    name = "twist_torque"
    summary = "constant torque about +z on one body (gel_tactile twist law)"

    @dataclass
    class Config:
        body: str = "box"
        torque: float = TWIST

    def setup(self, ctx: Any) -> None:
        self.body = ctx.bodies(self.config.body)

    def apply(self, state: newton.State, forces: Forces, dt: float) -> None:
        wp.launch(_torque, dim=self.ctx.world_count, inputs=[self.body, float(self.config.torque)],
                  outputs=[forces.body_f], device=self.ctx.device)


def _final(run: Any) -> dict[str, np.ndarray]:
    out = run.plugin.outputs()
    res = {k: out[k].numpy()[0, 0] for k in ("depth", "marker_disp", "force")}
    res["prm"] = run.plugin._prm.numpy()[:10]
    return res


def _check_cap(run: Any) -> LawResult:
    q = run.trace.body_q[-1]
    pad, ball = q[run.bodies("pad")[0]], q[run.bodies("ball")[0]]
    pen = float(pad[2] + R_BALL - ball[2])
    p = run.plugin
    d = _final(run)["depth"]
    xs = (np.arange(p.cols) + 0.5) * p.cell - p.cols * p.cell / 2
    ys = p.rows * p.cell / 2 - (np.arange(p.rows) + 0.5) * p.cell
    r = np.hypot(xs[None, :] - (ball[0] - pad[0]), ys[:, None] - (ball[1] - pad[1]))
    cap = np.clip(pen - (R_BALL - np.sqrt(np.clip(R_BALL ** 2 - r ** 2, 0.0, None))), 0.0, None)
    err = float(np.abs(d - cap).max()) / max(pen, 1e-12)
    return LawResult.within(err, 0.02, f"penetration {pen * 1e6:.2f} um, peak depth {d.max() * 1e6:.2f} um, "
                            f"max |depth - cap| {err * 100:.2f} % of it",
                            "the depth field must be the engine's penetration: check the gel frame and the SDF trace")


def _check_weight(run: Any) -> LawResult:
    m = float(run.model.body_mass.numpy()[run.bodies("ball")[0]])
    fn = float(_final(run)["force"][0])
    return LawResult.within((fn - m * 9.81) / (m * 9.81), 5e-3, f"gel normal {fn:.5g} N vs m·g {m * 9.81:.5g} N",
                            "the contact capture misses the gel contacts")


def _green_np(dx: np.ndarray, dy: np.ndarray, a: float, nu: float, table: np.ndarray | None,
              pix: float) -> np.ndarray:
    """[..., 3, 3] Green's tensors at offsets (dx, dy), as in gel_loads.green (numpy, float64)."""
    if table is not None:
        ix = 450 + np.rint(dx / pix).astype(int)
        iy = 450 - np.rint(dy / pix).astype(int)
        ok = (ix >= 0) & (ix <= 899) & (iy >= 0) & (iy <= 899)
        out = np.zeros(dx.shape + (3, 3))
        out[ok] = table[iy[ok], ix[ok]]
        return out
    r2 = dx * dx + dy * dy + a * a
    r = np.sqrt(r2)
    t, b, r3 = 2.0 * (1.0 - nu) / r, (1.0 - 2.0 * nu) / r2, r2 * r
    g = np.zeros(dx.shape + (3, 3))
    g[..., 0, 0], g[..., 0, 1], g[..., 0, 2] = t + 2 * nu * dx * dx / r3, 2 * nu * dx * dy / r3, -b * dx
    g[..., 1, 0], g[..., 1, 1], g[..., 1, 2] = 2 * nu * dx * dy / r3, t + 2 * nu * dy * dy / r3, -b * dy
    g[..., 2, 0], g[..., 2, 1], g[..., 2, 2] = b * dx, b * dy, t
    return g


def reference_markers(run: Any) -> np.ndarray:
    """World 0, sensor 0: the marker field recomputed in numpy from the final depth and the engine's force/twist."""
    p, f = run.plugin, _final(run)
    blk = p.config.block
    dsum = f["depth"].reshape(p.cr, blk, p.cc, blk).sum((1, 3)).reshape(-1).astype(np.float64)
    ci, cj = np.divmod(np.arange(p.cr * p.cc), p.cc)
    xy = np.stack([(cj + 0.5) * p.coarse - p.cc * p.coarse / 2, p.cr * p.coarse / 2 - (ci + 0.5) * p.coarse], -1)
    total = dsum.sum()
    if total <= 0.0 or f["prm"][0] <= 0.0:
        return np.zeros((len(p.marker_xy), 3))
    cbar = (dsum[:, None] * xy).sum(0) / total
    rel = xy - cbar
    r2 = (dsum * (rel ** 2).sum(1)).sum()
    fn, ftx, fty, tau = (float(v) for v in f["prm"][:4])
    w = dsum / total
    tw = tau * dsum / r2 if r2 > 0 else np.zeros_like(dsum)
    loads = np.stack([ftx * w - tw * rel[:, 1], fty * w + tw * rel[:, 0], fn * w], -1)
    table = p._table.numpy().reshape(900, 900, 3, 3) if p.config.kernel == "taxim" else None
    from alloy.plugins.tactile.taxim import FEM_PIXMM  # noqa: PLC0415

    def field(q: np.ndarray) -> np.ndarray:
        g = _green_np(q[:, None, 0] - xy[None, :, 0], q[:, None, 1] - xy[None, :, 1], p.coarse, p.config.poisson,
                      table, FEM_PIXMM)
        return np.einsum("mcij,cj->mi", g, loads)

    peak = int(np.argmax(dsum))
    scale = (dsum[peak] / blk ** 2) / field(xy[peak:peak + 1])[0, 2]
    u = scale * field(np.asarray(p.marker_xy, np.float64))
    return u * np.array([1.0, 1.0, -1.0])


def _check_markers(run: Any) -> LawResult:
    got = _final(run)["marker_disp"].astype(np.float64)
    want = reference_markers(run)
    ref = max(float(np.abs(want).max()), 1e-15)
    err = float(np.abs(got - want).max()) / ref
    return LawResult.within(err, 0.01, f"max marker error {err * 100:.3f} % of max |u| {ref * 1e6:.3f} um",
                            "the GPU loads or superposition disagree with the numpy model")


def _check_twist_sense(run: Any) -> LawResult:
    p, f = run.plugin, _final(run)
    q, u = np.asarray(p.marker_xy), f["marker_disp"]
    circ = float((q[:, 0] * u[:, 1] - q[:, 1] * u[:, 0]).sum())
    tau = float(f["prm"][3])
    bad = 0.0 if circ * tau > 0 else 1.0
    return LawResult.within(bad, 0.0, f"marker circulation {circ:.3g} m², gel twist {tau:.3g} N·m",
                            "the twist term must turn the markers with the moment on the gel")


def _check_null(run: Any) -> LawResult:
    f = _final(run)
    v = max(float(np.abs(f[k]).max()) for k in ("depth", "marker_disp", "force"))
    return LawResult.within(v, 0.0, f"max |depth, marker, force| {v:.3g} with nothing on the gel",
                            "a contact outside the capture depth reached the gel")


def gel_laws() -> list[Law]:
    """The laws above, on their scenes; markers on a 5 x 7 grid."""
    shear = TestScene(build=L1._build_rest, periods=25, ground_mu=1.0, gravity=L1._tilted(0.3, 30.0))
    twist = TestScene(build=L1._build_rest, periods=25, ground_mu=1.0, companions=lambda: [TwistTorque()])
    return [
        Law("depth_cap", _check_cap, doc="depth = the engine's spherical-cap penetration"),
        Law("force_weight", _check_weight, doc="gel normal force = m·g"),
        Law("markers_normal", _check_markers, config=MARKERS, doc="markers = numpy model, sphere press"),
        Law("markers_shear", _check_markers, scene=shear, config=MARKERS, doc="markers = numpy model, shear"),
        Law("markers_twist", _check_markers, scene=twist, config=MARKERS, doc="markers = numpy model, twist"),
        Law("twist_sense", _check_twist_sense, scene=twist, config=MARKERS, doc="markers turn with the twist"),
        Law("null", _check_null, scene=TestScene(build=L1._build_null, periods=10, ground_mu=1.0), config=MARKERS,
            doc="no contact, no field"),
    ]
