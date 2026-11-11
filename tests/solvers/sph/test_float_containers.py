# SPDX-License-Identifier: Apache-2.0
"""WP9 (a1) floating containers, routing through the public API (CPU, build only): a container flagged by its card
(``features.passage_width``) floating in a liquid routes like a floating solid up to D5's measured ratio bound, and
is a FloatingContainerError above it (the laws run on the GPU: test_float_containers_gpu.py)."""

from __future__ import annotations

import pytest

import alloy
from alloy.scene.coupler_select import FLOATING_CONTAINER_RATIO_MAX
from tests.solvers.sph.float_scenes import Float


def test_a_floating_container_below_the_bound_builds_and_routes_with_the_liquid():
    """A cup at D5 ratio 10 (and one at the bound) builds: the cup on the rigid entry, the water on SPH."""
    for ratio in (10.0, FLOATING_CONTAINER_RATIO_MAX):
        sim = alloy.build(Float(ratio).scene(), device="cpu")
        routes = {n: r["route"] for n, r in sim.routes().items()}
        assert routes["cup"] == "rigid" and routes["water"] == "liquid_sph"


def test_a_floating_container_beyond_the_bound_is_refused_with_its_fix():
    """At 2x the bound the build refuses it, naming the ratio, the bound and the mass that would route."""
    with pytest.raises(alloy.BuildError) as e:
        alloy.build(Float(2.0 * FLOATING_CONTAINER_RATIO_MAX).scene(), device="cpu")
    assert "floating container" in e.value.message and "mass >=" in e.value.message
    assert e.value.fix.startswith("make 'cup' at least ") and " kg (it is " in e.value.fix      # the exact mass


@pytest.mark.compile_heavy
def test_a_container_at_the_bound_steps_on_the_cpu():
    """WP4's ask: the same physics steps on the CPU at the bound (a small scene: a 40 mm cup at D5 ratio 50 in an
    80 mm tank of 2,000 particles, 0.1 s): it lands on the water and floats within one spacing (4 mm) of its
    Archimedes bottom (its 0.6 mm draft is under a particle radius here), nothing is non-finite, and the world is
    healthy."""
    import numpy as np  # noqa: PLC0415

    f = Float(FLOATING_CONTAINER_RATIO_MAX, L=0.04, H=0.03, tank=0.08, depth=0.02)
    sim = alloy.build(f.scene(), device="cpu")
    traj = sim.run(0.1, record=["cup"], every=1)
    z = traj["cup.pos"][:, 0, 2] - f.surface
    print(f"bottom {1e3 * z[-1]:+.2f} mm (min {1e3 * z.min():+.2f}), want {1e3 * f.bottom_expected():+.2f} mm; "
          f"check {sim.check()['ok']}")
    assert np.isfinite(traj["cup.pos"]).all() and abs(z[-1] - f.bottom_expected()) <= 4.0e-3 and sim.check()["ok"]
