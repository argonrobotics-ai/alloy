# SPDX-License-Identifier: Apache-2.0
"""The MuJoCo C CPU route cannot be captured as a graph, and says so (CPU; agent friction F3).

It steps MuJoCo C on the host, which a Warp CPU graph (a record-only op stream) cannot hold. Before, a capture failed
as "host read inside a captured control period (array.numpy() x4)"; now the route declares it
(``graph_capturable``) and a step during a CPU capture raises naming the warp CPU route, which captures.
"""

from __future__ import annotations

import pytest
import warp as wp

from alloy.core.errors import GraphCaptureError
from alloy.parallel.graphs import cpu_graphs_supported
from alloy.route.mjc_core import MjcCoreSolver, graph_capturable
from alloy.route.mjc_core_config import CPU_ROUTE_KEY
from test_mjc_core import DT, arm_model, factory, route

pytestmark = pytest.mark.cpu_route


def test_the_route_declares_it_cannot_be_captured_and_the_warp_route_can():
    model = arm_model()
    ok, why = graph_capturable(route(model))
    assert not ok and "alloy:cpu_route='warp'" in why and "ALLOY_CPU_ROUTE=warp" in why
    warp = factory(arm_model(), **{CPU_ROUTE_KEY: "warp"})
    assert not isinstance(warp, MjcCoreSolver) and graph_capturable(warp) == (True, "")


@pytest.mark.skipif(not cpu_graphs_supported(), reason="this Warp records no CPU graphs")
def test_a_step_inside_a_cpu_capture_names_the_warp_route():
    model = arm_model()
    solver = route(model)
    s0, s1, ctrl = model.state(), model.state(), model.control()
    solver.step(s0, s1, ctrl, None, DT)                  # builds the route outside any capture
    wp.capture_begin(device="cpu")
    try:
        with pytest.raises(GraphCaptureError, match="ALLOY_CPU_ROUTE=warp"):
            solver.step(s1, s0, ctrl, None, DT)
    finally:
        wp.capture_end(device="cpu")
    solver.step(s1, s0, ctrl, None, DT)                  # still steps eagerly afterwards
