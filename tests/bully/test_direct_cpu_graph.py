# SPDX-License-Identifier: Apache-2.0
"""DirectSim CPU graph replay (SCR-WP8-5 Ask 1, WP7 review): a period replays a captured pair of substeps.

- Bitwise equal to eager (state, MuJoCo Warp data, exported contacts, health records) on a two-box pile and on the
  routed T12 target (WP5's ordered route, Alloy's MuJoCo plug-ins).
- ``reset`` (all worlds and masked) recaptures and replays the same bits as eager; health records are restored.
- ``graph_safe`` hooks and observers run inside the capture; an unmarked one keeps the period eager.
- A new binding recaptures; a refused capture falls back to eager (``graph=None``) or raises (``graph=True``); odd
  substep counts and subclasses that override ``substep`` stay eager.
"""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from alloy._vendor import newton

from alloy.bully.direct import DirectSim, graph_safe
from alloy.core.errors import BullyError, GraphCaptureError

PERIODS = 4


def _pile(worlds: int) -> newton.Model:
    b = newton.ModelBuilder()
    for i in range(2):
        body = b.add_body(xform=wp.transform(wp.vec3(0.03 * i, 0.0, 0.05 + 0.099 * i), wp.quat_identity()))
        b.add_shape_box(body, hx=0.05, hy=0.05, hz=0.05)
    s = newton.ModelBuilder()
    s.replicate(b, worlds)
    s.add_ground_plane()
    return s.finalize(device="cpu")


def _sim(graph: bool | None = None, *, worlds: int = 2, substeps: int = 4, cls: type = DirectSim) -> DirectSim:
    model = _pile(worlds)
    solver = newton.solvers.SolverMuJoCo(model, njmax=200, nconmax=64, solver="newton")
    return cls(model, solver, control_dt=0.01, substeps=substeps, collide=False, export_contacts=True, graph=graph)


def _bits(sim: DirectSim) -> list[np.ndarray]:
    d, c = sim.solver.mjw_data, sim.contacts
    arrays = [sim.state.body_q, sim.state.body_qd, d.qpos, d.qvel, d.qacc_warmstart, c.rigid_contact_count,
              c.rigid_contact_point0, c.rigid_contact_normal, *sim.health.state_arrays().values()]
    return [a.numpy().copy() for a in arrays]


def _raw(x: np.ndarray) -> bytes:
    """The bytes of every field (a structured record's padding bytes are not data)."""
    if x.dtype.names is None:
        return x.tobytes()
    return b"".join(np.ascontiguousarray(x[name]).tobytes() for name in x.dtype.names)


def _same(a: list[np.ndarray], b: list[np.ndarray]) -> None:
    """Bit for bit (health records hold NaN fields, and -0.0 must not pass for 0.0)."""
    assert len(a) == len(b)
    for i, (x, y) in enumerate(zip(a, b, strict=True)):
        assert x.dtype == y.dtype and x.shape == y.shape and _raw(x) == _raw(y), f"array {i} differs"


def _run(sims: list[DirectSim], periods: int = PERIODS) -> None:
    for _ in range(periods):
        for s in sims:
            s.step_period()
        _same(_bits(sims[1]), _bits(sims[0]))


def test_graph_replay_is_bitwise_equal_to_eager():
    eager, graph = _sim(False), _sim()
    assert graph.eager_reason() is None and eager.eager_reason() == "graph=False"
    _run([eager, graph])
    assert graph.graph_mode == "graph" and graph.graph_captures == 1 and eager.graph_mode == "eager"
    assert (graph.clock.ticks, graph.periods) == (eager.clock.ticks, eager.periods) and graph.periods == PERIODS
    assert graph.clock.t == pytest.approx(PERIODS * 0.01)
    assert int(graph.contacts.rigid_contact_count.numpy()[0]) > 0  # exported MuJoCo contacts, every world in touch


def test_reset_and_health_restore_replay_the_same_bits():
    eager, graph = _sim(False), _sim()
    _run([eager, graph], 2)
    first = _bits(graph)
    for s in (eager, graph):
        s.reset()
    assert graph.graph_mode == "eager" and graph.periods == 0  # reset dropped the captured pair
    _run([eager, graph], 2)
    _same(_bits(graph), first)
    mask = np.array([0, 1], dtype=np.int32)
    for s in (eager, graph):
        s.reset(mask)
    _run([eager, graph], 2)
    assert graph.graph_captures == 3
    rep = graph.report()["health"]
    assert rep is not None and rep.flags.tolist() == [0, 0]


@wp.kernel
def _push(cmd: wp.array[wp.spatial_vector], body_f: wp.array[wp.spatial_vector]):
    i = wp.tid()
    body_f[i] = cmd[i]


@wp.kernel
def _peak_speed(qd: wp.array[wp.spatial_vector], peak: wp.array[wp.float32]):
    i = wp.tid()
    peak[i] = wp.max(peak[i], wp.length(wp.spatial_top(qd[i])))


class _Push:
    """A body-force hook from a device command (as HookedForceActions) and a peak-speed observer, both kernels only."""

    def __init__(self, sim: DirectSim) -> None:
        n = sim.model.body_count
        self.cmd = wp.zeros(n, dtype=wp.spatial_vector, device="cpu")
        self.peak = wp.zeros(n, dtype=wp.float32, device="cpu")
        sim.pre_substep.append(self.hook)
        sim.add_step_observer(self.observe)

    @graph_safe
    def hook(self, state: newton.State) -> None:
        wp.launch(_push, dim=self.cmd.shape[0], inputs=[self.cmd, state.body_f], device="cpu")

    @graph_safe
    def observe(self, s0, s1, control, contacts, dt) -> None:
        wp.launch(_peak_speed, dim=self.peak.shape[0], inputs=[s1.body_qd, self.peak], device="cpu")


def test_marked_hooks_and_observers_run_inside_the_capture():
    eager, graph = _sim(False), _sim()
    pushes = [_Push(eager), _Push(graph)]
    for p in range(PERIODS):
        f = np.zeros((eager.model.body_count, 6), np.float32)
        f[:, 0] = 2.0 * (p + 1)  # a new command each period, written outside the captured pair
        for push in pushes:
            push.cmd.assign(f)
        _run([eager, graph], 1)
        np.testing.assert_array_equal(pushes[1].peak.numpy(), pushes[0].peak.numpy())
    assert graph.graph_mode == "graph" and graph.graph_captures == 1
    assert float(pushes[1].peak.numpy().max()) > 0.0


def test_unmarked_hook_keeps_the_period_eager():
    sim, calls = _sim(), []

    def count(state):  # Python state per call: a replay would not run it
        calls.append(1)

    sim.pre_substep.append(count)
    sim.step_period()
    assert "count" in sim.eager_reason() and sim.graph_mode == "eager" and len(calls) == 4
    sim.pre_substep.remove(count)
    sim.step_period()
    sim.step_period()
    assert sim.eager_reason() is None and sim.graph_mode == "graph" and len(calls) == 4


def test_new_binding_recaptures():
    eager, graph = _sim(False), _sim()
    _run([eager, graph], 2)
    assert graph.graph_captures == 1
    _Push(eager), _Push(graph)
    _run([eager, graph], 2)
    assert graph.graph_captures == 2


def test_refused_capture_falls_back_to_eager():
    eager, sim = _sim(False), _sim()

    @graph_safe
    def reads_host(state):  # marked, but reads the host: the CPU capture's guard refuses it
        state.body_q.numpy()

    for s in (eager, sim):
        s.pre_substep.append(reads_host)
    _run([eager, sim], 1)  # the first pair is eager (warm-up), the second capture is refused and runs eagerly
    assert "capture refused" in sim.eager_reason() and "host read" in sim.eager_reason()
    assert sim.graph_mode == "eager" and sim.clock.ticks == eager.clock.ticks
    _run([eager, sim], 2)


def test_refused_capture_raises_when_asked_for():
    sim = _sim(True)

    @graph_safe
    def reads_host(state):
        state.body_q.numpy()

    sim.pre_substep.append(reads_host)
    with pytest.raises(GraphCaptureError, match="host read"):
        sim.step_period()


class _Override(DirectSim):
    def substep(self) -> None:
        super().substep()


def test_odd_substeps_and_overrides_stay_eager():
    assert "one substep" in _sim(substeps=1).eager_reason()
    with pytest.raises(BullyError, match="pairs"):
        _sim(True, substeps=1)
    sim = _sim(cls=_Override)
    assert "overrides substep" in sim.eager_reason()
    sim.step_period()
    assert sim.graph_mode == "eager" and sim.clock.t == pytest.approx(0.01)


@pytest.mark.compile_heavy
@pytest.mark.timeout(180)
def test_t12_target_replays_bitwise():
    """The routed T12 target (ordered route, scoped noslip, coordinate carry, line contacts, contact export): two
    ticks of B0-like controls from reset, eager then graph, in one sim."""
    from alloy.bully.targets import get_target

    sim = get_target("task_t12").make_sim(1, "cpu")
    assert sim.eager_reason() is None
    rng = np.random.default_rng(0)
    ctrl = sim.control.mujoco.ctrl.numpy()
    rows = [ctrl + rng.normal(0.0, 0.02, ctrl.shape).astype(np.float32) for _ in range(3)]
    out = {}
    for flag in (False, None):
        sim.graph = flag
        sim.reset()
        for row in rows:
            sim.control.mujoco.ctrl.assign(row)
            sim.step_period()
        out[flag] = _bits(sim)
    assert sim.graph_mode == "graph"
    _same(out[None], out[False])
