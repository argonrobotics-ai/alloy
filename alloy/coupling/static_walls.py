# SPDX-License-Identifier: Apache-2.0
"""Static shapes are walls for particle entries in a coupled scene (SCR-JOB2-5, coordinator ruling 2026-11-05 00:31Z).

A static shape has no dynamics to own: it is world-attached (body -1) or sits on a world-welded body (an incoming
FIXED joint to the world or to another welded body, ``alloy.coupling.kinematic.world_welded_bodies``; a robot's bench
base, say). In a single-solver scene every particle solver sees it. In a coupled
scene Newton's entry views hide every shape the entry neither owns nor sees through a proxy
(``SolverCoupled._apply_entry_shape_visibility``): world shapes stay visible by default (``_entry_visible_shapes``
adds body -1's shapes when the entry declares none), but a shape on a welded base body of the MuJoCo entry vanishes
from the SPH view, so a spill fell through JOB2's bench plywood.

``static_wall_shapes`` lists the shapes on welded static bodies that collide with particles in the parent model.
``SchedulerMixin`` adds them to every particle entry's visible shapes (the Newton hook that feeds both the collision
mask and the view compaction) and then keeps only their particle-collision bit (``narrow_static_walls``): a wall for
particles, never a rigid collider or an owned shape in that view (``Entry.shapes`` would imply ownership, and two
particle entries cannot own the same shape). Dynamic shapes keep Newton's rule: the owner sees them, or they are seen
through a proxy. A shape whose body is outside the entry's selection makes Newton build that view at the full model
layout (its compaction refuses such a shape); the static body is then present with its dynamics disabled.

The two hooks are private Newton methods (``SolverCoupled._entry_visible_shapes``, ``_customize_compact_view``);
``tests/integration/test_static_walls.py`` checks that Newton still defines them, so a rename fails loudly instead of
silently dropping the walls; both are listed in ``alloy.core.newton_compat.PRIVATE_HOOKS`` (WP0, main 3b464ed4).
"""

from __future__ import annotations

import logging
from typing import Any

import warp as wp

from alloy._vendor import newton
from alloy.coupling.kinematic import world_welded_bodies

log = logging.getLogger("alloy.coupling")

__all__ = ["narrow_static_walls", "static_wall_shapes"]

_RIGID_BITS = int(newton.ShapeFlags.COLLIDE_SHAPES) | int(newton.ShapeFlags.HYDROELASTIC)


def static_wall_shapes(model: Any) -> list[int]:
    """Shapes on world-welded bodies (``alloy.coupling.kinematic.world_welded_bodies``, the one definition WP4's
    full-surface records and kinematic proxies use; not body -1, which Newton already shows) that collide with
    particles."""
    if model.shape_count == 0 or model.shape_flags is None:
        return []
    welded = world_welded_bodies(model)
    if not welded.any():
        return []
    body, flags = model.shape_body.numpy(), model.shape_flags.numpy()
    bit = int(newton.ShapeFlags.COLLIDE_PARTICLES)
    return [s for s in range(model.shape_count) if int(body[s]) >= 0 and welded[int(body[s])] and int(flags[s]) & bit]


@wp.kernel(enable_backward=False)
def _clear_bits_kernel(shapes: wp.array[int], bits: int, flags: wp.array[int]):
    s = shapes[wp.tid()]
    flags[s] = flags[s] & ~bits


def narrow_static_walls(view: Any, shapes: list[int]) -> None:
    """Keep only the particle-collision bit of ``shapes`` in ``view`` (global shape ids: Newton never compacts
    shapes), so a particle entry never treats a static wall as a rigid collider of its own proxies. Writes a copy
    and assigns it (``ModelView``'s public override), so the parent's flags are untouched."""
    if not shapes:
        return
    flags = wp.clone(view.shape_flags)
    wp.launch(_clear_bits_kernel, dim=len(shapes), inputs=[wp.array(shapes, dtype=int, device=flags.device),
                                                            _RIGID_BITS], outputs=[flags], device=flags.device)
    view.shape_flags = flags
