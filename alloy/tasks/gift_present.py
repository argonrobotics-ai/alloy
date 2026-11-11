# SPDX-License-Identifier: Apache-2.0
"""The wrapped present (library card ``gift_wrapped_present``) on its box, for scenes (docs/notes/paper-pack.md).

The card is the paper only (alloy/tasks/gift_wrap_mesh.py); the box inside is a rigid box of the wrap's size whose
bottom rests on the paper's bottom layers. The lowest layer (the seam's outer one) lies one contact radius plus a pitch
below the box, and a film particle collides at its contact radius, so the present stands 2.2 mm up off its surface.

    scene = alloy.Scene(dt=0.01); scene.add_ground()
    wrap, box = add_present(scene, "present")
"""

from __future__ import annotations

from dataclasses import dataclass

from alloy.solvers.film.materials import FilmExtras

from .gift_wrap_mesh import WrapSpec

CARD = "gift_wrapped_present"
WRAP = WrapSpec()                       # the card's mesh
LIFT = 1.0e-4                           # start gap under the lowest particle [m]


@dataclass(frozen=True)
class PresentSpec:
    box_mass: float = 0.45              # ESTIMATE: a chipboard gift box with a small present inside [kg]
    friction: float = 0.4               # ESTIMATE: kraft on chipboard (Coulomb)


def lowest_layer(w: WrapSpec = WRAP) -> float:
    """Depth of the paper's lowest mid-surface below the box: the seam's outer layer [m]."""
    return w.r + w.pitch


def add_present(scene, name: str = "present", xy=(0.0, 0.0), z: float = 0.0,
                spec: PresentSpec = PresentSpec()) -> tuple[str, str]:
    """Add the wrap ``name`` and its box ``name + "_box"`` resting on a surface at height ``z``; returns both names."""
    z0 = z + lowest_layer() + FilmExtras().contact_radius + LIFT          # the box's bottom
    x, y = (float(v) for v in xy)
    L, W, H = WRAP.box
    box = f"{name}_box"
    scene.add_object(name, CARD, pos=(x, y, z0))
    scene.add_box(box, size=(L, W, H), pos=(x, y, z0 + 0.5 * H), mass=spec.box_mass, friction=spec.friction)
    return name, box
