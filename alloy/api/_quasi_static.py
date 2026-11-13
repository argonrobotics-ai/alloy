# SPDX-License-Identifier: Apache-2.0
"""``Scene.set_quasi_static_mass_scale``: opt-in quasi-static mass scaling of a j2 solid on the MPM route (GAP #6 slice
2b step 2; physics ruling 2026-11-10; the physics in ``alloy.solvers.plastic.quasi_static``).

Build order (``alloy.api._build.build_block``):
1. ``with_scales``: the router reads ``scene.quasi_static`` and gives each named j2 solid its ``Scale`` (s, the
   predicted inertial share; ``sim.routes()`` names both).
2. ``apply``, after assembly and before wiring: refuses (SceneError) an object that is not a j2 solid on the MPM route,
   a share over ``MAX_SHARE``, or a solid that does not rest on the ground or a static box; then multiplies its particle
   masses by s, so the MPM solver's cached density and the health monitor's energy scale both see them.
3. ``plugin_for``: a force plug-in (``QuasiStaticWeight``) that adds (s - 1) m0 g upward on each scaled particle every
   exchange (the weight stays true; its work counts as outside work) and measures, per world, the solid's mean
   velocity and the fastest body touching it against the declared tool speed; ``register`` hands both to the health
   monitor (QUASI_STATIC, DEGRADED).
A scene without the setting builds exactly as before: no router note, no scaling, no plug-in, no health check.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from alloy.api.errors import SceneError

__all__ = ["QuasiStaticMixin", "apply", "plugin_for", "register", "with_scales"]

_FIX_ROUTE = ("give it a plastic.model j2 material and the object card's route override paste_mpm "
              "(card_extra={'route': {'route': 'paste_mpm', 'reason': ...}}), or drop set_quasi_static_mass_scale")


class QuasiStaticMixin:
    """``Scene.set_quasi_static_mass_scale``: the setting lives on the object's item (``items[name]["quasi_static"]``:
    {"scale", "tool_speed"}), so ``Scene.to_dict`` / ``from_dict`` carry it with the item; ``quasi_static`` reads
    them all (name -> setting; empty without any)."""

    items: dict[str, dict[str, Any]]

    @property
    def quasi_static(self) -> dict[str, dict[str, Any]]:
        """Every object's quasi-static setting: name -> {"scale", "tool_speed"} (a copy)."""
        return {n: dict(it["quasi_static"]) for n, it in self.items.items() if it.get("quasi_static")}

    def set_quasi_static_mass_scale(self, name: str, scale: Any = "auto", *, tool_speed: float) -> None:
        """Quasi-static mass scaling of a j2 solid on the MPM route (a block dented by a tool): its density times s,
        its weight kept true, so a stiff card's stress solve converges (brass: 1.00 of its converged punch force at
        x4,000-x16,000 against 0.007 unscaled). Quasi-static only: the solid is s times heavier for anything that
        moves it, so the build refuses a solid that does not rest on the ground or a static box, and it is wrong
        for dynamics (a drop, a throw, a fast hit). ``tool_speed`` (m/s): the fastest a tool or robot presses on it.
        ``scale="auto"`` takes the largest s whose inertial share v sqrt(E rho s) / yield stays within 1 %; a number
        over 1 % warns, over 5 % is refused. ``sim.routes()[name]`` names s and the share; a world whose solid then
        moves faster reads DEGRADED with QUASI_STATIC. Example:
        ``scene.set_quasi_static_mass_scale("block", tool_speed=0.002)``."""
        if name not in self.items:
            raise SceneError(f"set_quasi_static_mass_scale: no object named {name!r}",
                             f"one of {sorted(self.items)}", field="name")
        if scale != "auto" and not (isinstance(scale, (int, float)) and math.isfinite(scale) and scale >= 1.0):
            raise SceneError(f"set_quasi_static_mass_scale({name!r}): scale must be 'auto' or a number >= 1, got "
                             f"{scale!r}", "scale='auto'", field="scale")
        if not (isinstance(tool_speed, (int, float)) and math.isfinite(tool_speed) and tool_speed > 0.0):
            raise SceneError(f"set_quasi_static_mass_scale({name!r}): tool_speed must be a positive speed in m/s, got "
                             f"{tool_speed!r}", "the fastest a tool presses on it, e.g. tool_speed=0.002",
                             field="tool_speed")
        self.items[name]["quasi_static"] = {"scale": scale if scale == "auto" else float(scale),
                                            "tool_speed": float(tool_speed)}


def with_scales(router: Any, scene: Any) -> Any:
    """``router`` with ``scene.quasi_static`` handed to the base router (read by ``Router._quasi_static_scale``)."""
    if not getattr(scene, "quasi_static", None):
        return router
    base = router
    while hasattr(base, "router"):                  # ClampRouter, fixture routers: the Router they wrap
        base = base.router
    base.quasi_static = dict(scene.quasi_static)
    return router


def _rows(built: Any, name: str) -> np.ndarray:
    """[worlds, n] global particle rows of object ``name`` (``alloy.scene.randomize``'s layout)."""
    h = built.handles[name]
    idx = np.asarray(list(h.template.particle), np.int64)
    worlds = np.arange(int(built.model.world_count), dtype=np.int64)[:, None]
    return worlds * len(h.world_stride.particle) + idx[None, :]


def _supports(scene: Any) -> list[tuple[str, float, np.ndarray | None]]:
    """(name, top z, xy footprint [[x0, y0], [x1, y1]] or None for an infinite plane) of the ground and static boxes."""
    out = []
    for name, it in scene.items.items():
        prim, pose = (it.get("shape") or {}).get("primitive") or {}, it.get("pose")
        if not it.get("static") or pose is None:
            continue
        if it["kind"] == "ground":
            out.append((name, float(pose[2]), None))
        elif prim.get("kind") == "box":
            from scipy.spatial.transform import Rotation  # noqa: PLC0415

            half = np.abs(Rotation.from_quat(pose[3:7]).as_matrix()) @ np.asarray(prim["size"], float)  # AABB half
            c = np.asarray(pose[:3], float)
            out.append((name, float(c[2] + half[2]), np.stack([c[:2] - half[:2], c[:2] + half[:2]])))
    return out


def _check_support(built: Any, scene: Any, name: str, rows: np.ndarray, voxel: float) -> None:
    """SceneError unless, in every world, the solid's lowest particle is within a voxel of a support's top and its
    centre is over that support (the ground plane, or a static box)."""
    q = built.model.particle_q.numpy()[rows]                     # [W, n, 3]
    bottom, centre = q[..., 2].min(axis=1), q[..., :2].mean(axis=1)
    sup = _supports(scene)
    def over(w: int, fp: np.ndarray | None) -> bool:
        return fp is None or bool(np.all((centre[w] >= fp[0]) & (centre[w] <= fp[1])))

    for w in range(rows.shape[0]):
        if any(abs(bottom[w] - top) <= voxel and over(w, fp) for _, top, fp in sup):
            continue
        raise SceneError(f"{name}: quasi-static mass scaling needs a solid that rests on the ground or on a static box "
                         f"(world {w}: its bottom is at z {bottom[w]:.4f} m, no support top within one voxel, "
                         f"{voxel * 1e3:g} mm). A scaled solid is s times heavier for anything that moves it, so a "
                         "free one would fall, slide or bounce wrong", "rest it on the ground or a static box, or drop "
                         "set_quasi_static_mass_scale", field="quasi_static")


def apply(built: Any, scene: Any) -> None:
    """Refuse what the setting cannot do, then scale each named solid's particle masses (module docstring, step 2)."""
    from alloy.solvers.plastic import quasi_static as qs  # noqa: PLC0415

    for name in getattr(scene, "quasi_static", None) or {}:
        d = built.handles[name].decision
        if qs.NUMBER not in d.numbers:
            raise SceneError(f"{name}: quasi-static mass scaling takes a j2 solid on the MPM route; {name} routed "
                             f"{d.route}", _FIX_ROUTE, field="quasi_static")
        s, share = float(d.numbers[qs.NUMBER]), float(d.numbers[qs.SHARE_NUMBER])
        if share > qs.MAX_SHARE:
            v_mm = d.numbers[qs.SPEED_NUMBER] * 1e3
            raise SceneError(f"{name}: quasi-static mass scaling x{s:,.0f} at {v_mm:g} mm/s has an inertial share of "
                             f"{share * 100:.2g} %, over {qs.MAX_SHARE * 100:g} %: the solid's scaled inertia would "
                             "push back on the tool", "scale='auto', a smaller scale, or a slower tool_speed",
                             field="quasi_static")
        rows = _rows(built, name)
        _check_support(built, scene, name, rows, float(d.numbers["voxel"]))
        if s == 1.0:
            continue
        m = built.model
        flat = rows.ravel()
        mass, inv = m.particle_mass.numpy(), m.particle_inv_mass.numpy()
        mass[flat] *= s
        inv[flat] = np.where(inv[flat] > 0.0, inv[flat] / s, 0.0)
        m.particle_mass.assign(mass)
        m.particle_inv_mass.assign(inv)


def plugin_for(scene: Any) -> Any:
    """A function of the built scene that makes the weight-and-share plug-in, or None without the setting."""
    if not getattr(scene, "quasi_static", None):
        return None
    names = tuple(scene.quasi_static)

    def make(built: Any) -> Any:
        from alloy.solvers.plastic import quasi_static as qs  # noqa: PLC0415
        from alloy.solvers.plastic.quasi_static_plugin import QuasiStaticWeight, Solid  # noqa: PLC0415

        solids = []
        for name in names:
            n = built.handles[name].decision.numbers
            solids.append(Solid(name, _rows(built, name), float(n[qs.NUMBER]), float(n[qs.SPEED_NUMBER]),
                                float(n["voxel"])))
        return QuasiStaticWeight(solids=solids)
    return make


def _mpm_solver(engine: Any) -> Any:
    """The engine's MPM solver (the one whose grid reports collider impulses), or None."""
    for solver in getattr(engine, "solvers", ()) or ():
        if hasattr(solver, "collect_collider_impulses") and hasattr(solver, "collider_body_index"):
            return solver
    return None


def register(engine: Any) -> None:
    """Give the plug-in the MPM solver (its contact speeds) and hand its per-world motion and contact ratios to the
    health monitor (QUASI_STATIC over ``MOTION_MARGIN`` / ``CONTACT_MARGIN``); a no-op without the plug-in."""
    from alloy.solvers.plastic import quasi_static as qs  # noqa: PLC0415
    from alloy.solvers.plastic.quasi_static_plugin import QuasiStaticWeight  # noqa: PLC0415

    host = getattr(engine, "plugins", None)
    for plugin in getattr(host, "plugins", {}).values() if host is not None else ():
        if not isinstance(plugin, QuasiStaticWeight):
            continue
        plugin.attach(_mpm_solver(engine))
        for k, solid in enumerate(plugin.solids):
            what = f"quasi-static mass scaling of {solid.name}"
            engine.health.register_share_check(f"{what}: motion", plugin.motion[k], qs.MOTION_MARGIN)
            engine.health.register_share_check(f"{what}: contact", plugin.contact[k], qs.CONTACT_MARGIN)
