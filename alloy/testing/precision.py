# SPDX-License-Identifier: Apache-2.0
"""WP5 test helpers: an eager health loop with injection hooks, and a scripted gap monitor (SPEC §6.1, D17).

``HealthLoop`` runs collide -> solver.step -> inject -> gap.update -> health.update -> health.latch -> swap on a
``RefScene``; ``inject(step, state_out)`` stands in for a solver or contact layer producing a bad value, so PRC-9
(recall) and PRC-10 (false alarms) can compare an injected run with a clean one bit for bit.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence

import numpy as np
import warp as wp

from alloy._vendor import newton
from alloy.core.snapshot import WorldSnapshot
from alloy.precision.health import HealthMonitor
from alloy.precision.types import CheckPolicy, PrecisionConfig, StepHealth
from alloy.testing.scenes import RefScene

log = logging.getLogger("alloy.testing")

Injector = Callable[[int, newton.State], None]


@wp.kernel
def _write_pair_pen(values: wp.array[wp.float32], bits: wp.array[wp.uint32], health: wp.array[StepHealth]):
    w = wp.tid()
    h = health[w]
    h.max_pair_pen_m = wp.max(h.max_pair_pen_m, values[w])
    h.flags = h.flags | bits[w]
    health[w] = h


class ScriptedGapMonitor:
    """``GapMonitorLike`` that writes a scripted per-world ``max_pair_pen_m`` (max, like WP3's GapMonitor) and
    scripted flag bits (OR) into the health inbox (D11)."""

    def __init__(self, world_count: int, device) -> None:
        self.values = wp.zeros(world_count, dtype=wp.float32, device=device)
        self.bits = wp.zeros(world_count, dtype=wp.uint32, device=device)
        self.updates = 0

    def set_bits(self, world: int, bits: int) -> None:
        host = self.bits.numpy()
        host[world] = bits
        self.bits.assign(host)

    def set(self, world: int, pen_m: float) -> None:
        host = self.values.numpy()
        host[world] = pen_m
        self.values.assign(host)

    def update(self, state_prev: newton.State, state: newton.State, health: wp.array | None = None) -> None:
        self.updates += 1
        if health is not None:
            wp.launch(_write_pair_pen, dim=self.values.shape[0], inputs=[self.values, self.bits, health],
                      device=self.values.device)

    def report(self):  # pragma: no cover - not used by the WP5 tests
        from alloy.contact.types import GapReport

        return GapReport.empty(0, self.values.shape[0])


class HealthLoop:
    """Eager §2.3 sub-loop for health tests on one ``RefScene``."""

    def __init__(self, scene: RefScene, *, checks: CheckPolicy | None = None, kind: str | None = None,
                 config: PrecisionConfig | None = None, capacity_flags: Sequence = (), precision_pairs=(),
                 energy: str = "auto", solver_kwargs: dict | None = None) -> None:
        self.scene = scene
        self.runner = scene.runner(kind, **(solver_kwargs or {}))
        model = scene.model
        self.gap = ScriptedGapMonitor(scene.world_count, model.device)
        self.monitor = HealthMonitor(model, checks or CheckPolicy(level="full", full_every=1),
                                     precision_pairs=precision_pairs, capacity_flags=capacity_flags,
                                     solvers=[self.runner.solver], config=config, energy=energy)
        self.snapshot = WorldSnapshot(model, self.runner.state_0)
        self.snapshot.save(self.runner.state_0)
        self.steps = 0
        self.history: list[np.ndarray] = []

    @property
    def state(self) -> newton.State:
        return self.runner.state_0

    def step(self, n: int = 1, inject: Injector | None = None, record: bool = False) -> None:
        r = self.runner
        for _ in range(n):
            if self.steps == 0 or self.monitor.is_check_step(self.steps - 1):
                self.snapshot.save(r.state_0)  # last good = output of the last checked step
            r.state_0.clear_forces()
            if r.pipeline is not None:
                r.pipeline.collide(r.state_0, r.contacts)
            r.solver.step(r.state_0, r.state_1, r.control, r.contacts, self.scene.dt)
            if inject is not None:
                inject(self.steps, r.state_1)
            self.gap.update(r.state_0, r.state_1, self.monitor.health)
            self.monitor.update(self.steps, r.state_0, r.state_1, r.contacts)
            self.monitor.latch(r.state_1, self.snapshot)
            if record:
                self.history.append(self.monitor.record.numpy().copy())
            r.state_0, r.state_1 = r.state_1, r.state_0
            self.steps += 1

    def body_state(self) -> tuple[np.ndarray, np.ndarray]:
        return self.state.body_q.numpy().copy(), self.state.body_qd.numpy().copy()


def set_body(state_arr: wp.array, body: int, value) -> None:
    """Host write of one body's row (test injection)."""
    host = state_arr.numpy()
    host[body] = value
    state_arr.assign(host)


def world_bodies(model: newton.Model, world: int) -> np.ndarray:
    return np.flatnonzero(model.body_world.numpy() == world)


def fake_sim(scene_name: str, n: int, kind: str | None = None, device: str | None = None):
    """A ``FakeSimulation`` over a reference scene (until WP2's Simulation lands): XPBD box dropped 5 cm onto the
    plane (real Newton contact pipeline, impact then rest), Featherstone pendulum without contacts."""
    from alloy.testing import scenes
    from alloy.testing.fakes import FakeContactLayer, FakeSimulation, FakeStepper

    scene = scenes.box_on_plane(n, gap=0.05, device=device) if scene_name == "box_on_plane" else \
        scenes.pendulum(n, 0.4, device=device)
    solver = scene.make_solver(kind)
    layer = FakeContactLayer(scene.model, real_pipeline=True) if scene_name == "box_on_plane" else None
    return FakeSimulation(scene.model, FakeStepper(solver, dt_x=scene.dt), contact_layer=layer)


def box_sim(n: int):
    """Module-level ``make_sim`` (picklable for R2): XPBD box on a plane, Warp CPU."""
    return fake_sim("box_on_plane", n, device="cpu")


def pendulum_sim(n: int):
    """Module-level ``make_sim`` (picklable for R2): Featherstone pendulum, Warp CPU."""
    return fake_sim("pendulum", n, device="cpu")


# ----- GPU repeat scenes (PRC-4/5/6/13/14) -----

def box_drop(n: int, *, boxes: int = 27, device: str | None = None, seed: int = 7, nudge_worlds: bool = False
             ) -> newton.Model:
    """``boxes`` 5 cm cubes per world dropped from a jittered 3-wide lattice onto a shared ground plane (precision
    research §3.3 probe). Worlds are identical unless ``nudge_worlds`` (``nudge_one_ulp``)."""
    rng = np.random.default_rng(seed)
    template = newton.ModelBuilder()
    for k in range(boxes):
        ix, iy, iz = k % 3, (k // 3) % 3, k // 9
        p = np.array([0.12 * (ix - 1), 0.12 * (iy - 1), 0.08 + 0.12 * iz]) + rng.uniform(-0.01, 0.01, 3)
        b = template.add_body(xform=wp.transform(wp.vec3(*p), wp.quat_identity()))
        template.add_shape_box(b, hx=0.05, hy=0.05, hz=0.05)
    root = newton.ModelBuilder()
    root.replicate(template, n)
    root.add_ground_plane()
    model = root.finalize(device=device)
    if nudge_worlds:
        nudge_one_ulp(model, boxes)
    return model


def nudge_one_ulp(model: newton.Model, boxes: int) -> None:
    """Seeded 1-ulp perturbations (PRC-14 reference): world w >= 1 moves one coordinate of one body by exactly one
    ulp (body (w-1) % boxes, axis ((w-1) // boxes) % 3, sign alternating per sweep); world 0 is untouched."""
    q = model.body_q.numpy()
    world = model.body_world.numpy()
    for w in range(1, int(model.world_count)):
        bodies = np.flatnonzero(world == w)
        k = w - 1
        b, axis = bodies[k % boxes], (k // boxes) % 3
        toward = np.float32(np.inf) if (k // (3 * boxes)) % 2 == 0 else np.float32(-np.inf)
        q[b, axis] = np.nextafter(q[b, axis], toward)
    model.body_q.assign(q)
    model.joint_q.assign(_joint_q_from_bodies(model, q))


def _joint_q_from_bodies(model: newton.Model, body_q: np.ndarray) -> np.ndarray:
    """Free-joint coordinates (p, q) from body poses, for models whose bodies all hang on free joints."""
    jq = model.joint_q.numpy()
    starts, child = model.joint_q_start.numpy(), model.joint_child.numpy()
    for j in range(model.joint_count):
        jq[starts[j]:starts[j] + 7] = body_q[child[j]]
    return jq


class DetSim:
    """Minimal ``SimulationLike`` for repeat tests: [collide +] solver.step per substep; ``use_graph`` captures one
    control period as a CUDA graph (an even substep count keeps the state buffers in place)."""

    def __init__(self, model: newton.Model, solver, *, pipeline=None, dt: float = 0.02 / 16, control_dt: float = 0.02,
                 use_graph: bool = False) -> None:
        from alloy.core.timebase import SimClock, grid_index

        self.model, self.solver, self.pipeline = model, solver, pipeline
        self.state, self._next, self.control = model.state(), model.state(), model.control()
        newton.eval_fk(model, model.joint_q, model.joint_qd, self.state)
        self.contacts = pipeline.contacts() if pipeline is not None else None
        self.clock = SimClock(control_dt, grid_index(dt, control_dt))
        if self.clock.steps_per_period % 2:
            raise ValueError("DetSim needs an even number of substeps per period")
        self.use_graph = bool(use_graph) and wp.get_device(model.device).is_cuda
        self._graph = None
        self.solvers = (solver,) + ((pipeline,) if pipeline is not None else ())
        self.precision = None

    def _period(self) -> None:
        for _ in range(self.clock.steps_per_period):
            self.state.clear_forces()
            if self.pipeline is not None:
                self.pipeline.collide(self.state, self.contacts)
            self.solver.step(self.state, self._next, self.control, self.contacts, self.clock.dt)
            self.state, self._next = self._next, self.state

    def step_period(self) -> None:
        if not self.use_graph:
            self._period()
        else:
            if self._graph is None:
                with wp.ScopedCapture(device=self.model.device) as cap:
                    self._period()
                self._graph = cap.graph
            wp.capture_launch(self._graph)
        self.clock.advance(self.clock.steps_per_period)

    def reset(self, world_mask: wp.array | None = None) -> None:
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state)

    def report(self) -> dict:
        return {"t": self.clock.t}

    def state_arrays(self) -> dict:
        from alloy.core.registry import state_arrays_for

        _register_state_adapters()
        return dict(state_arrays_for(self.solver))

    def state_world_maps(self) -> dict:
        from alloy.core.registry import state_world_maps_for

        _register_state_adapters()
        return dict(state_world_maps_for(self.solver))


def _register_state_adapters() -> None:
    """Register the Newton solvers' state adapters (SolverMuJoCo: warm start, qacc, cvel, ...). ``alloy.build``
    registers them, but a DetSim drives raw solvers. Before this, its snapshots held State only unless something
    had imported alloy.coupling first, and R1 diverged at period 1. PRC-13's MuJoCo 'requested' case run alone
    failed that way on main and on the Newton bump (2026-11-09)."""
    from alloy.coupling.adapters import register_builtin_adapters

    register_builtin_adapters()


def det_sim(solver_kind: str, n: int, *, det: bool = True, use_graph: bool = False, boxes: int = 27,
            nudge_worlds: bool = False, sorted_contacts: bool | None = None,
            precision: PrecisionConfig | None = None) -> DetSim:
    """A ``box_drop`` DetSim on the default device. ``solver_kind``: xpbd, steel, mujoco, featherstone, fake.
    ``det`` turns on the solver's deterministic mode and (unless ``sorted_contacts`` says otherwise) a sorted Newton
    contact pipeline. SolverMuJoCo uses Alloy's MuJoCo settings (Newton solver, implicitfast, elliptic cone) with a
    sparse Jacobian. With ``precision``, ``enable_solver_determinism`` runs on the built solver (Alloy's ordered
    SolverMuJoCo route under a RUN_TO_RUN preset)."""
    model = box_drop(n, boxes=boxes, nudge_worlds=nudge_worlds)
    mode = wp.DeterministicMode.RUN_TO_RUN if det else wp.DeterministicMode.NOT_GUARANTEED
    if solver_kind == "fake":
        from alloy.testing.core import FakeSolver

        return DetSim(model, FakeSolver(model, FakeSolver.Config(body_radius=0.05)), use_graph=use_graph)
    factories = {
        "xpbd": lambda: newton.solvers.SolverXPBD(model, deterministic=mode),
        # 27 piled boxes request ~2,000 contact/friction rows per world; Steel's default 512 drops rows (ROW_OVERFLOW)
        "steel": lambda: newton.solvers.SolverSteel(model, deterministic=mode, free_body_constraint_max=96 * boxes),
        "featherstone": lambda: newton.solvers.SolverFeatherstone(model),
        "mujoco": lambda: newton.solvers.SolverMuJoCo(model, deterministic=mode, use_mujoco_contacts=False,
                                                      nconmax=64 * boxes, njmax=256 * boxes, solver="newton",
                                                      integrator="implicitfast", cone="elliptic", jacobian="sparse"),
    }
    solver = factories[solver_kind]()
    if precision is not None:
        from alloy.precision.determinism import enable_solver_determinism

        enable_solver_determinism(precision, [solver])
    pipeline = newton.CollisionPipeline(model, deterministic=det if sorted_contacts is None else sorted_contacts)
    return DetSim(model, solver, pipeline=pipeline, use_graph=use_graph)


def write_row(metric: str, value, verdict: str, command: str, **extra) -> None:
    """Append one result row to ``bench/results/wp5_results.jsonl`` (copied back by ``gpu_run.sh``)."""
    import json
    import os
    import subprocess
    import time
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    out = root / "bench" / "results"
    out.mkdir(parents=True, exist_ok=True)
    commit = os.environ.get("ALLOY_COMMIT") or subprocess.run(
        ["git", "rev-parse", "--short=12", "HEAD"], cwd=root, capture_output=True, text=True, check=False).stdout.strip()
    row = {"metric": metric, "value": value, "verdict": verdict, "command": command, "commit": commit or "unknown",
           "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **extra}
    with (out / "wp5_results.jsonl").open("a") as fh:
        fh.write(json.dumps(row, default=str) + "\n")


def resting_boxes(n: int, layout: str, *, device: str | None = None) -> newton.Model:
    """PRC-1 rest scenes: ``rest-16`` (a 4x4 grid of 5 cm cubes on the plane) or ``3-stack`` (three cubes stacked),
    placed in contact, 1 kg each."""
    half = 0.025
    template = newton.ModelBuilder()
    cfg = newton.ModelBuilder.ShapeConfig(density=1.0 / (8.0 * half**3), mu=0.5)
    if layout == "rest-16":
        spots = [(0.08 * (i - 1.5), 0.08 * (j - 1.5), half) for i in range(4) for j in range(4)]
    elif layout == "3-stack":
        spots = [(0.0, 0.0, half * (2 * k + 1)) for k in range(3)]
    else:
        raise ValueError(f"unknown rest layout {layout!r}")
    for p in spots:
        b = template.add_body(xform=wp.transform(wp.vec3(*p), wp.quat_identity()))
        template.add_shape_box(b, hx=half, hy=half, hz=half, cfg=cfg)
    root = newton.ModelBuilder()
    root.replicate(template, n)
    root.add_ground_plane()
    return root.finalize(device=device)


# ----- MuJoCo Warp contact / constraint-row state (P3-1 route c tests) -----

def mjw_contacts(d) -> dict:
    """Host copies of the used contact slots (every per-slot field plus efc_address)."""
    from alloy.precision import mjw_order

    n = int(d.nacon.numpy()[0])
    return {f: getattr(d.contact, f).numpy()[:n].copy() for f in (*mjw_order._contact_fields(d), "efc_address")}


def mjw_rows_state(d) -> dict:
    """Host copies of each world's used constraint rows, its J storage and its JTDAJ blocks."""
    from alloy.precision import mjw_rows

    e, nefc = d.efc, d.nefc.numpy()
    out = {f: [getattr(e, f).numpy()[w, :nefc[w]].copy() for w in range(len(nefc))] for f in mjw_rows._fields(True)
           if getattr(e, f).shape[1] > 0}
    if e.J_rownnz.shape[1] > 0:
        nnz = [int(e.J_rownnz.numpy()[w, :nefc[w]].sum()) for w in range(len(nefc))]
        out["J"] = [e.J.numpy()[w, 0, :nnz[w]].copy() for w in range(len(nefc))]
        out["J_colind"] = [e.J_colind.numpy()[w, 0, :nnz[w]].copy() for w in range(len(nefc))]
    else:
        out["J"] = [e.J.numpy()[w, :nefc[w]].copy() for w in range(len(nefc))]
    nb = e.jtdaj_nblock.numpy()
    out["jtdaj"] = [np.stack([e.jtdaj_adr.numpy()[w, :nb[w]], e.jtdaj_nrow.numpy()[w, :nb[w]]]) for w in range(len(nb))]
    return out


def shuffle_mjw_contacts(d, rng, imported: bool = False) -> None:
    """Reorder the used contact slots as another counter order would. ``imported``: the slots were filled before
    make_constraint (Newton contacts), so ``efc_address`` moves with them and contact row ids are remapped."""
    from alloy.precision import mjw_order

    n = int(d.nacon.numpy()[0])
    p = rng.permutation(n)  # new slot i holds old slot p[i]
    fields = [*mjw_order._contact_fields(d), *(["efc_address"] if imported else [])]
    for f in fields:
        a = getattr(d.contact, f).numpy()
        a[:n] = a[p]
        getattr(d.contact, f).assign(a)
    if imported:
        inv = np.empty(n, dtype=int)
        inv[p] = np.arange(n)
        rtype, rid, nefc = d.efc.type.numpy(), d.efc.id.numpy(), d.nefc.numpy()
        for w in range(len(nefc)):
            rows = np.flatnonzero(rtype[w, :nefc[w]] >= 5)  # contact rows
            rid[w, rows] = inv[rid[w, rows]]
        d.efc.id.assign(rid)


def block_permutation(rtype: np.ndarray, rid: np.ndarray, rng) -> np.ndarray:
    """A random order of the row blocks (runs of equal (type, id)); rows inside a block keep their order, as one
    thread writes them consecutively after one counter reservation."""
    starts = [0] + [i for i in range(1, len(rtype)) if (rtype[i], rid[i]) != (rtype[i - 1], rid[i - 1])]
    blocks = [np.arange(a, b) for a, b in zip(starts, [*starts[1:], len(rtype)])]
    return np.concatenate([blocks[i] for i in rng.permutation(len(blocks))]) if blocks else np.zeros(0, int)


def shuffle_mjw_rows(d, rng) -> None:
    """Permute each world's row blocks; sparse row content stays in its storage slot (rowadr moves with the row)."""
    from alloy.precision import mjw_rows

    e, nefc = d.efc, d.nefc.numpy()
    host = {f: getattr(e, f).numpy() for f in mjw_rows._fields(True) if getattr(e, f).shape[1] > 0}
    dense_j = e.J.numpy() if e.J_rownnz.shape[1] == 0 else None
    adr, nrow, nb = e.jtdaj_adr.numpy(), e.jtdaj_nrow.numpy(), e.jtdaj_nblock.numpy()
    eaddr, cw = d.contact.efc_address.numpy(), d.contact.worldid.numpy()
    for w in range(len(nefc)):
        n = int(nefc[w])
        p = block_permutation(host["type"][w, :n], host["id"][w, :n], rng)  # new row r holds old row p[r]
        inv = np.empty(n, dtype=int)
        inv[p] = np.arange(n)
        for a in host.values():
            a[w, :n] = a[w, p]
        if dense_j is not None:
            dense_j[w, :n] = dense_j[w, p]
        adr[w, :nb[w]] = inv[adr[w, :nb[w]]]
        order = rng.permutation(int(nb[w]))
        adr[w, :nb[w]], nrow[w, :nb[w]] = adr[w, order], nrow[w, order]
        mine = (cw == w)[:, None] & (eaddr >= 0)
        eaddr[mine] = inv[eaddr[mine]]
    for f, a in host.items():
        getattr(e, f).assign(a)
    if dense_j is not None:
        e.J.assign(dense_j)
    e.jtdaj_adr.assign(adr)
    e.jtdaj_nrow.assign(nrow)
    d.contact.efc_address.assign(eaddr)


def assert_mjw_equal(a: dict, b: dict) -> None:
    for k in a:
        if isinstance(a[k], list):
            for x, y in zip(a[k], b[k]):
                assert np.array_equal(x, y), k
        else:
            assert np.array_equal(a[k], b[k]), k


def probe_npz(tmp, name: str, module: str, *flags: str, timeout: int = 2400) -> tuple[dict, dict]:
    """Run ``python -m <module> ... --out <tmp>/<name>.npz`` in a fresh process: (npz arrays, meta)."""
    import json
    import os
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    out = Path(tmp) / f"{name}.npz"
    env = {**os.environ, "PYTHONPATH": str(root), "OMP_NUM_THREADS": "4"}
    proc = subprocess.run([sys.executable, "-m", module, *flags, "--out", str(out)], cwd=root,
                          env=env, capture_output=True, text=True, timeout=timeout, check=False)
    assert proc.returncode in (0, 3) and out.exists(), proc.stderr[-2000:]
    z = np.load(out)
    return {k: z[k] for k in z.files if k != "meta"}, json.loads(str(z["meta"]))


def probe_run(tmp, name: str, *flags: str):
    """Run ``python -m alloy.tools.mjw_det_probe`` in a fresh process: (digests, meta, eager_s)."""
    arrays, meta = probe_npz(tmp, name, "alloy.tools.mjw_det_probe", *flags)
    return arrays["digests"], meta, arrays["eager_s"]


def compare_digests(a: np.ndarray, b: np.ndarray) -> tuple[float, int, float]:
    """(share of worlds equal after the last call, first call with any difference or -1, equal share over all)."""
    eq = (a == b).all(axis=-1)
    first = -1 if eq.all() else int(np.argmax(~eq.all(axis=1)))
    return float(eq[-1].mean()), first, float(eq.mean())


def repeat_verdict(share: float, first: int) -> str:
    return "pass" if share == 1.0 and first == -1 else "fail"
