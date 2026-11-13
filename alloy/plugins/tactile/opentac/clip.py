# SPDX-License-Identifier: Apache-2.0
"""Render the gripper example (``example.py``) as an mp4: the scene from the front, each pad's ADC counts, and the
force traces (the engine's pad force against what the decoder reads back from the counts).

Needs only NumPy, matplotlib and ffmpeg (no Alloy), so it runs in any Python that has them::

    python alloy/plugins/tactile/opentac/clip.py opentac_grip.npz opentac_grip.mp4
"""

from __future__ import annotations

import sys

import numpy as np

PHASE_NAMES = ("close", "lift", "hold", "loosen", "slide")
GEOM = {"obj": (7.0, 15.0, 9.0), "pad": (12.0, 4.0, 10.5), "stand": 5.0}      # mm, as example.Gripper


def _traces(d: dict) -> dict:
    """Per period, summed over both pads: engine normal and shear, decoded normal and shear (N)."""
    pf = d["pad_force"]                                   # [N, P, 3]: normal, shear x, shear y (pad y = world up)
    return {"normal": pf[:, :, 0].sum(1), "shear": -pf[:, :, 2].sum(1),
            "dec_normal": d["decoded_taxel_n"].sum((1, 2)), "dec_shear": -d["decoded_shear"][..., 1].sum((1, 2)),
            "true_bump": -(d["bump_wrench"][..., 2] + 2 * d["bump_wrench"][..., 4] / d["bump_height"]).sum((1, 2))}


def _scene(ax, d: dict, k: int) -> None:
    import matplotlib.patches as mp  # noqa: PLC0415

    ax.clear()
    ox, oy, oz = GEOM["obj"]
    px, py, pz = GEOM["pad"]
    s = GEOM["stand"]
    z_obj = d["obj_q"][k, 2] * 1e3
    fz = 2 * s + oz + d["lift"][k] * 1e3
    gap = 4.0 - d["inward"][k] * 1e3
    ax.add_patch(mp.Rectangle((-40, -4), 80, 4, color="#bbbbbb"))
    ax.add_patch(mp.Rectangle((-s, 0), 2 * s, 2 * s, color="#999999"))
    ax.add_patch(mp.Rectangle((-oy, z_obj - oz), 2 * oy, 2 * oz, color="#d9a441"))
    for sgn in (-1, 1):
        y0 = sgn * (oy + py + max(gap, 0.0))
        ax.add_patch(mp.Rectangle((y0 - py, fz - pz), 2 * py, 2 * pz, color="#4a6fa5"))
        ax.add_patch(mp.Rectangle((y0 - sgn * py - 0.6 * (sgn > 0), fz - pz), 0.6, 2 * pz, color="#e4572e"))
    ax.set_xlim(-32, 32), ax.set_ylim(-4, 50), ax.set_aspect("equal"), ax.axis("off")
    ax.set_title(f"t = {k * float(d['control_dt']):.2f} s   {PHASE_NAMES[int(d['phase'][k])]}", fontsize=10)


def _pad(ax, d: dict, k: int, p: int, vmax: float) -> None:
    import matplotlib.patches as mp  # noqa: PLC0415

    ax.clear()
    xy = d["taxel_xy"] * 1e3
    c = d["counts"][k, p]
    ax.scatter(xy[:, 0], xy[:, 1], c=c, cmap="magma", vmin=0, vmax=vmax, marker="s", s=140)
    for bxy in d["bump_xy"] * 1e3:
        ax.add_patch(mp.Circle(bxy, 2.4, fill=False, lw=0.6, ec="#88c0d0"))
    ax.set_aspect("equal"), ax.set_xticks([]), ax.set_yticks([])
    ax.set_title(f"{('left', 'right')[p]} pad: counts (max {c.max():.0f})", fontsize=9)


def _plot_traces(ax, tr: dict, k: int, dt: float) -> None:
    ax.clear()
    t = np.arange(len(tr["normal"])) * dt
    ax.plot(t[:k + 1], tr["normal"][:k + 1], color="#4a6fa5", label="grip, engine (N)")
    ax.plot(t[:k + 1], tr["dec_normal"][:k + 1], "--", color="#4a6fa5", label="grip, decoded from counts")
    ax.plot(t[:k + 1], tr["shear"][:k + 1], color="#e4572e", label="up shear, engine")
    ax.plot(t[:k + 1], tr["dec_shear"][:k + 1], "--", color="#e4572e", label="up shear, decoded")
    ax.set_xlim(0, t[-1]), ax.set_ylim(min(-0.5, tr["dec_shear"].min() * 1.1), max(tr["normal"].max(),
                                                                                   tr["dec_shear"].max()) * 1.15)
    ax.set_xlabel("s", fontsize=8), ax.tick_params(labelsize=7), ax.legend(fontsize=7, loc="upper right")


def render(npz: str, out: str, fps: int = 8) -> None:
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415
    from matplotlib.animation import FFMpegWriter  # noqa: PLC0415

    d = dict(np.load(npz))
    tr, vmax = _traces(d), float(max(d["counts"].max(), 1.0))
    fig = plt.figure(figsize=(10, 5.6), dpi=110)
    g = fig.add_gridspec(2, 3, height_ratios=(1.25, 1))
    ax_scene, ax_l, ax_r = fig.add_subplot(g[0, 0]), fig.add_subplot(g[0, 1]), fig.add_subplot(g[0, 2])
    ax_t = fig.add_subplot(g[1, :])
    fig.suptitle("OpenTac pads in Alloy: what each board reads (OpenTac's own model, not checked against a real skin)",
                 fontsize=10)
    fig.text(0.5, 0.005, "Decoded shear reads low while held: one box corner sits on a 2-taxel bump block, which "
             "cannot sense vertical shear (an OpenTac board limit).", ha="center", fontsize=7, color="#555555")
    writer = FFMpegWriter(fps=fps, bitrate=1800)
    with writer.saving(fig, out, dpi=110):
        for k in range(len(d["phase"])):
            _scene(ax_scene, d, k)
            _pad(ax_l, d, k, 0, vmax)
            _pad(ax_r, d, k, 1, vmax)
            _plot_traces(ax_t, tr, k, float(d["control_dt"]))
            writer.grab_frame()
    plt.close(fig)


if __name__ == "__main__":
    render(sys.argv[1], sys.argv[2])
