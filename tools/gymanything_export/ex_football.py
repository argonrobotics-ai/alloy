# SPDX-License-Identifier: Apache-2.0
"""football: the regulation ball (thin spherical shell) and the goal frame (posts, crossbar and the training-goal
support rails), from ``football_specs.py`` and the collider list in ``newton15/world.py``; the cord net is not
exported (a many-node tension-only graph, not a single cable)."""

from __future__ import annotations

import ast
import inspect
import math

import numpy as np
import trimesh
from _common import Exporter, cited, cylinder, dim, line_of, revolve, union, vis, where

from football import football_specs as FS
from football.seam_geometry import PANELS
from football.newton15 import render as RN
from football.newton15 import world as W

BALL = FS.BallSpec()
FIELD = FS.FieldSpec()


def _srgb(lin) -> list[float]:
    """Their materials are UsdPreviewSurface diffuseColor values (linear); the part colour is sRGB."""
    return [12.92 * c if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055 for c in lin]


def _render_materials() -> dict:
    """The (colour, roughness, metallic) rows of the material tuple literal in their ``render.export``."""
    out = {}
    for node in ast.walk(ast.parse(inspect.getsource(RN.export).lstrip())):
        if isinstance(node, ast.Tuple) and len(node.elts) == 4 and isinstance(node.elts[0], ast.Constant):
            name, color, rough, metal = (ast.literal_eval(e) for e in node.elts)
            out[name] = (color, rough, metal)
    return out


def _look(geom, key: str, name: str) -> dict:
    rgb, rough, metal = _render_materials()[key]
    return vis(geom, _srgb(rgb), line_of(RN, f'("{key}", ('), rough, metal, name=f"{name} (linear->sRGB)")


def _ball_parts(shell) -> list[dict]:
    """Pentagons dark, hexagons white (their ``ball_panels``): each shell face goes to the nearest panel centre."""
    centres = np.array([p.center for p in PANELS], float)
    centres /= np.linalg.norm(centres, axis=1, keepdims=True)
    fc = shell.vertices[shell.faces].mean(axis=1)
    fc /= np.linalg.norm(fc, axis=1, keepdims=True)
    kind = np.array([p.kind for p in PANELS])[np.argmax(fc @ centres.T, axis=1)]
    parts = []
    for k, material in (("pentagon", "dark"), ("hexagon", "white")):
        sel = kind == k
        if sel.any():
            parts.append(_look(trimesh.Trimesh(shell.vertices, shell.faces[sel], process=False), material,
                               f"{k} panels"))
    return parts


def _param(name: str, quote: str = "") -> dict:
    p = FS.PARAMETERS[name]
    return cited(p.value, "standard", line_of(FS, f'"{name}": ParameterEvidence('), quote or f"{p.source}. {p.note}")


def _ball(x: Exporter) -> None:
    r, t = BALL.radius_m, BALL.shell_thickness_m
    n = 48
    ang = np.linspace(0, math.pi, n)
    outer = [(r * math.sin(a), r * math.cos(a)) for a in ang]
    inner = [((r - t) * math.sin(a), (r - t) * math.cos(a)) for a in ang[::-1]]
    poly = [p for p in outer[:-1]] + [(0.0, -r), (0.0, -(r - t))] + inner[1:-1] + [(0.0, r - t), (0.0, r)]
    # poles are on the axis: outer arc from the top to the bottom, then the inner arc back
    m = revolve(poly[:-1], 96)
    x.add("ball", "Regulation football (thin shell)", "other", m, visual=_ball_parts(m),
          material={"name": "synthetic leather panels, inflated"},
          mass=_param("ball_mass"), friction=cited(BALL.static_friction_prior, "estimate",
                                                   line_of(FS, "static_friction_prior"), "static friction prior 0.55"),
          restitution=cited(BALL.restitution_prior, "estimate", line_of(FS, "restitution_prior"),
                            "restitution prior 0.72"),
          friction_partner="grass pitch",
          dims=[dim("diameter", BALL.diameter_m, "max", line_of(FS, '"ball_circumference"'))],
          source_ref=where(FS.BallSpec),
          notes="Sphere of IFAB Law 2 circumference (0.69 m) as a 3 mm shell (their shell thickness); the gas inside "
                "is not modelled. The 32-panel seam grooves (1.5 mm) are not meshed.")


def _goal(x: Exporter) -> None:
    w, h = FIELD.goal_inside_width_m, FIELD.goal_inside_height_m
    rad = FIELD.goal_post_diameter_m / 2
    parts = [cylinder(rad, 1.22, (0, sy * 3.72, 1.22)) for sy in (-1, 1)]
    parts.append(cylinder(rad, 3.78, (0, 0, 2.50), (math.sin(math.pi / 4), 0, 0, math.cos(math.pi / 4))))
    goal = union(parts)
    goal.apply_translation(-goal.bounds.mean(axis=0) * [0, 1, 0])
    x.add("goal_frame", "Football goal frame (posts and crossbar)", "fixture", goal, motion="static",
          visual=[_look(goal, "white", "goal frame")],
          material={"name": "steel tube"},
          friction=cited(0.62, "authored", line_of(W, "default_shape_cfg.mu = 0.62"), "default shape mu 0.62"),
          friction_partner="ball",
          dims=[dim("inside width", w, "y", line_of(FS, '"goal_inside_width"')),
                dim("overall height", 2.50 + rad, "z", line_of(W, 'label="crossbar"'))],
          source_ref=line_of(W, 'label=f"goal_post_{y}"'),
          notes="Two 120 mm posts and the crossbar as their cylinder colliders (centres 7.44 m apart, so a 7.32 m gap "
                "inside the 0.06 m radius), unioned; the rear stays and ground rails of the training goal are not "
                "included. Fixed in place, no mass.")


def export(x: Exporter) -> None:
    for fn in (_ball, _goal):
        try:
            fn(x)
        except Exception as exc:  # noqa: BLE001
            x.fail(fn.__name__, f"{type(exc).__name__}: {exc}")
    x.fail("cord_net", "ball-catching net is a patch graph of ~0.1 m cords (many nodes), not one cable; not exported")
