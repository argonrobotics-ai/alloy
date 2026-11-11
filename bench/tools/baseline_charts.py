# SPDX-License-Identifier: Apache-2.0
"""Charts for bench/BASELINE.md (one takeaway each), written to bench/charts/.

    python -m bench.tools.baseline_charts       # needs matplotlib (not in the Newton venv)
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from bench.tools.baseline import BENCH, FAMILY_ORDER, build, load_family  # noqa: E402

OUT = os.path.join(BENCH, "charts")
# Reference categorical palette (light surface), fixed order; colour follows the solver everywhere.
SOLVER_COLOR = {
    "mujoco": "#2a78d6", "xpbd": "#eb6834", "steel": "#1baf7a", "kamino": "#eda100",
    "vbd": "#e87ba4", "mujoco_cg": "#6250d6",
}
GOOD, CRITICAL = "#0ca30c", "#d03b3b"
INK, INK2, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7", "#fcfcfb"


def _style(ax, title: str, subtitle: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.figure.text(0.02, 0.965, title, fontsize=10.5, weight="bold", color=INK, va="top")
    ax.figure.text(0.02, 0.895, subtitle, fontsize=8, color=INK2, va="top")


def _fig(left: float = 0.12):
    fig, ax = plt.subplots(figsize=(6, 3.5), dpi=150)
    fig.subplots_adjust(left=left, right=0.78, top=0.80, bottom=0.15)
    return fig, ax


def _worlds_axis(ax) -> None:
    ax.set_xscale("log", base=2)
    ticks = [1, 16, 256, 1024, 4096]
    ax.set_xticks(ticks, [str(t) for t in ticks])
    ax.minorticks_off()


def _end_labels(ax, ends: list[tuple[float, float, str]], log_y: bool, min_gap: float) -> None:
    """Direct labels at line ends, pushed apart so they never overlap (gap in axis units or decades)."""
    import math

    tf = (lambda v: math.log10(max(v, 1e-12))) if log_y else (lambda v: v)
    inv = (lambda v: 10**v) if log_y else (lambda v: v)
    placed: list[float] = []
    for x, y, text in sorted(ends, key=lambda e: tf(e[1])):
        yy = tf(y)
        if placed and yy - placed[-1] < min_gap:
            yy = placed[-1] + min_gap
        placed.append(yy)
        ax.annotate(text, (x, y), xytext=(x * 1.25, inv(yy)), textcoords="data", fontsize=8, color=INK,
                    va="center", annotation_clip=False)


def _save(fig, name: str) -> str:
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path


def _ladder(rows, scenario: str, config: str, value) -> list[tuple[int, float]]:
    pts = [(r["nworld"], value(r)) for r in rows
           if r["scenario"] == scenario and r["config"] == config and r.get("status") == "ok" and r.get("stable")]
    return sorted((n, v) for n, v in pts if v is not None)


def chart_pile_throughput() -> str:
    """S03 64-body bin: world-steps/s vs worlds per solver (graph on)."""
    rows = load_family("scale")
    fig, ax = _fig()
    ends: list[tuple[float, float, str]] = []
    for solver in ("xpbd", "steel", "kamino", "mujoco"):
        pts = _ladder(rows, "S03_throughput", f"bin64_{solver}_graph", lambda r: r.get("world_steps_per_s"))
        if not pts:
            continue
        xs, ys = zip(*pts)
        ax.plot(xs, ys, color=SOLVER_COLOR[solver], lw=2, marker="o", ms=4)
        ends.append((xs[-1], ys[-1], f"{solver}  {ys[-1]:,.0f}"))
    ax.set_yscale("log")
    _worlds_axis(ax)
    _end_labels(ax, ends, log_y=True, min_gap=0.24)
    ax.set_xlabel("worlds", color=INK2, fontsize=8)
    ax.set_ylabel("world-steps / s", color=INK2, fontsize=8)
    _style(ax, "64-body bin: every solver flattens by 256-1024 worlds",
           "S03 on one L4, CUDA graphs on, dt 1/60 s x 10 substeps. Featherstone goes NaN (not shown).")
    return _save(fig, "pile_throughput_vs_worlds.png")


def chart_law_pass_by_family(all_s) -> str:
    """Configs whose law row passes vs fails, per family."""
    fams, passed, failed = [], [], []
    for fam in FAMILY_ORDER:
        judged = [c for cfgs in all_s[fam].values() for c in cfgs.values() if c.law_pass is not None]
        if fam == "scale" or not judged:
            continue  # scale laws are determinism/graph checks, not physics accuracy
        fams.append(fam)
        passed.append(sum(c.law_pass for c in judged))
        failed.append(sum(not c.law_pass for c in judged))
    fig, ax = _fig()
    y = range(len(fams))[::-1]
    ax.barh(list(y), passed, color=GOOD, height=0.55, edgecolor=SURFACE, linewidth=2)
    ax.barh(list(y), failed, left=passed, color=CRITICAL, height=0.55, edgecolor=SURFACE, linewidth=2)
    for yi, p, f in zip(y, passed, failed):
        ax.text(p + f + 0.4, yi, f"{p} of {p + f} pass", va="center", fontsize=8, color=INK)
    ax.set_xlim(0, max(p + f for p, f in zip(passed, failed)) * 1.4)
    ax.set_yticks(list(y), fams)
    ax.set_xlabel("stock Newton configs (L4 law row)", color=INK2, fontsize=8)
    ax.grid(axis="y", visible=False)
    ax.legend(["pass", "fail"], loc="lower right", fontsize=7, frameon=False)
    _style(ax, "Most stock configs fail their physics law outside rigid",
           "Law row = smallest full-horizon rung that covers every condition; scale family excluded.")
    return _save(fig, "law_pass_by_family.png")


def chart_peg_penetration(all_s) -> str:
    """P01 0.2 mm peg: penetration p99 per config against the 0.02 mm bound."""
    cfgs = all_s["precision"]["P01_peg_hole_0p2"]
    items = sorted(((k, v.penetration_p99_mm) for k, v in cfgs.items() if v.penetration_p99_mm), key=lambda t: t[1])
    bound = 0.02
    fig, ax = _fig(left=0.2)
    names = [k for k, _ in items]
    vals = [v for _, v in items]
    colors = [CRITICAL if v > bound else SOLVER_COLOR["mujoco"] for v in vals]
    ax.barh(names, vals, color=colors, height=0.55)
    ax.axvline(bound, color=INK2, lw=1, ls="--")
    ax.text(bound, len(names) - 0.4, " bound 0.02 mm", fontsize=7, color=INK2, va="bottom")
    for i, v in enumerate(vals):
        ax.text(v * 1.15, i, f"{v * 1000:.2g} µm" if v < 0.01 else f"{v:.3f} mm", va="center", fontsize=8, color=INK)
    ax.set_xscale("log")
    ax.set_xlabel("penetration p99 (mm, log)", color=INK2, fontsize=8)
    ax.grid(axis="y", visible=False)
    _style(ax, "Peg in 0.2 mm hole: only sdf_mujoco sinks past the p99 bound",
           "P01, max over the 1-4096 world ladder. All six still fail the tilt-excess part of the law.")
    return _save(fig, "peg_hole_penetration_p99.png")


def chart_tower_collapse() -> str:
    """R04 box tower: share of worlds whose tower collapsed vs batch size."""
    rows = load_family("rigid")
    series = (("mujoco_cg", "mujoco_cg", "MuJoCo CG"), ("vbd", "vbd", "VBD"),
              ("mujoco_newton", "mujoco", "MuJoCo Newton"), ("steel", "steel", "Steel"), ("xpbd", "xpbd", "XPBD"))
    fig, ax = _fig()
    zero: list[str] = []
    for cfg, color_key, label in series:
        pts = _ladder(rows, "R04_box_stack_10", cfg,
                      lambda r: None if (r.get("metrics") or {}).get("worlds_collapsed") is None
                      else 100.0 * r["metrics"]["worlds_collapsed"] / r["nworld"])
        if not pts:
            continue
        xs, ys = zip(*pts)
        ax.plot(xs, ys, color=SOLVER_COLOR[color_key], lw=2, marker="o", ms=4)
        if max(ys) == 0:
            zero.append(label)
        else:
            ax.annotate(f"{label}  {ys[-1]:.0f} %", (xs[-1], ys[-1]), xytext=(6, 0), textcoords="offset points",
                        fontsize=8, color=INK, va="center")
    if zero:
        ax.annotate("\n".join(zero) + "\n0 % at every size", (4096, 0), xytext=(6, 14), textcoords="offset points",
                    fontsize=8, color=INK, va="bottom")
    _worlds_axis(ax)
    ax.set_ylim(-4, 104)
    ax.set_xlabel("worlds (identical towers)", color=INK2, fontsize=8)
    ax.set_ylabel("towers collapsed (%)", color=INK2, fontsize=8)
    _style(ax, "Box tower: MuJoCo CG collapses more the more worlds run",
           "R04 10-cube tower, 10 s rest on one L4. Every world is the same scene; 1 world stands.")
    return _save(fig, "tower_collapse_vs_worlds.png")


def main() -> None:
    all_s = build()
    for p in (chart_pile_throughput(), chart_law_pass_by_family(all_s), chart_peg_penetration(all_s),
              chart_tower_collapse()):
        print(p)


if __name__ == "__main__":
    main()
