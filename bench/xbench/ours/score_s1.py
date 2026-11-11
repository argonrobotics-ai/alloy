# SPDX-License-Identifier: Apache-2.0
"""Score a T12 / embodiment trajectory npz (any engine) with Alloy's own SB-7 checker (Alloy venv).

    PYTHONPATH=. python bench/xbench/ours/score_s1.py run.npz [--device cuda:0] [--out run_s1.json]

The frames are written into Newton states of the model Alloy's SB-7 target builds (``meta.alloy_target``, default
``task_t12``: the same shapes, hulls and margins; its mesh geoms collide as convex hulls, as MuJoCo collides them,
``hull_meshes.py``), contacts come from Newton's ``CollisionPipeline`` on the frame before each step
(``DirectSim(collide=True)`` order: rows detected on ``state_in``, evaluated at ``state_out``; ``--touch union``
also takes robot touch from the frame after the step), and every pair of consecutive frames goes through ``BenchMonitors.update``
(``PipelineMonitors``, which Alloy's own beside runs use in-process too, so every engine has one scorer). Severity is ``alloy.bully.severity.classify`` with no
health flags (Alloy's SB-7 runs ``DirectSim``, whose report carries none) and no solver overflow (an engine's
capacity overflow is reported by its runner, not seen here). So the S1 rows (nonfinite, teleport, max_speed,
impossible_speed, pen_s1_ratio, e_gain_cum) are evaluated by Alloy's code on the engine's poses.

Frames should be every physics step (``record_every`` 1): teleport and the energy ledger are per step, and contact
touch decides the impossible-speed excuse per step.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import warp as wp

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[2]), str(HERE)]

from hull_meshes import colliding_mesh_shapes  # noqa: E402
from packs_io import DEFAULT_TARGET, mujoco_shape_pairs, target_model  # noqa: E402

log = logging.getLogger("xbench.ours")

# Pooled Newton-pipeline budgets per world. Newton's own estimates (T12 81k contact rows and 18k triangle pairs per
# world, H1-2 152k / 134k) do not fit 2,048 worlds on an L4; contact rows are 4x the targets' MuJoCo Warp nconmax.
# A pass that overflows any of them (rows, triangle pairs, the contact reducer) grows the budgets and is scored again
# (``PipelineMonitors.needed``), so rows are never dropped silently.
CONTACTS_PER_WORLD = 1024
TRIANGLE_PAIRS_PER_WORLD = 4096
_MIN_TRIANGLE_PAIRS = 1 << 20


@wp.kernel
def _buffer_peak_kernel(rows: wp.array[wp.int32], triangles: wp.array[wp.int32], ht_failures: wp.array[wp.int32],
                        reducer_overflows: wp.array[wp.int32], peak: wp.array[wp.int32]):
    wp.atomic_max(peak, 0, rows[0])
    wp.atomic_max(peak, 1, triangles[0])
    wp.atomic_max(peak, 2, ht_failures[0] + reducer_overflows[0])


def _narrow_counters(narrow, device) -> tuple:
    """(triangle pairs asked for, reducer hash-table failures, reducer buffer overflows), each a 1-element int32
    array; Newton leaves a counter None when the scene has nothing that needs it (primitive-only Go2: no mesh, no
    reducer), so those read a zero instead of a null pointer."""
    zero = wp.zeros(1, dtype=wp.int32, device=device)
    red = getattr(narrow, "global_contact_reducer", None)
    found = (getattr(narrow, "triangle_pairs_count", None), getattr(red, "ht_insert_failures", None),
             getattr(red, "buffer_overflows", None))
    return tuple(zero if c is None else c for c in found)


def load(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def _collision_model(model, hulls):
    """The model the scorer's pipeline collides: ``hulls`` (``target_model``: mesh geoms as convex hulls) beside a live
    sim's ``model``, else ``model``. A live model's state drives ``hulls``'s pipeline, so the shapes must match."""
    m = model if hulls is None else hulls
    if hulls is not None and (hulls.shape_count != model.shape_count
                              or not np.array_equal(hulls.shape_body.numpy(), model.shape_body.numpy())):
        raise ValueError(f"collision model ({hulls.shape_count} shapes) does not match the sim's model "
                         f"({model.shape_count} shapes) shape for shape")
    left = colliding_mesh_shapes(m).size
    if left:
        log.warning("scorer: %d colliding shapes are triangle meshes, not convex hulls as MuJoCo collides them "
                    "(Newton's mesh-mesh path; build the model with packs_io.target_model)", left)
    return m


def _order(names: list[str], model_names: list[str]) -> np.ndarray:
    """Index into the npz B axis for each Newton body (by name); raises if a body is missing."""
    pos = {n: i for i, n in enumerate(names)}
    missing = [n for n in model_names if n not in pos]
    if missing:
        raise ValueError(f"npz lacks bodies {missing}")
    return np.array([pos[n] for n in model_names])


class PipelineMonitors:
    """Alloy's ``BenchMonitors`` with contacts from Newton's ``CollisionPipeline``: the one S1 path for every engine.

    ``touch="pre"`` collides the frame before the step (``DirectSim(collide=True)`` order, the default; contact rows
    and penetration come from it). ``"union"`` also collides the frame after the step and ORs its touch flags in
    (``touch_any``, ``touch_robot``, ``world_robot_touch``; no extra penetration rows), so a robot contact present at
    either end of a coarse step excuses it (WP7: an e_gain / impossible_speed S1 counts only if the union pass trips
    it too; the union only adds excuses). Also a ``DirectSim`` step observer, so Alloy's own episodes go through the
    same code in-process: there ``collision_model`` is ``target_model``'s twin of the live model, whose mesh geoms
    collide as convex hulls (``hull_meshes``; the live sim keeps its own), and the live states drive its pipeline."""

    def __init__(self, model, touch: str = "pre", target: str = DEFAULT_TARGET,
                 contacts_per_world: int = CONTACTS_PER_WORLD, collision_model=None) -> None:
        from alloy.bully.monitors import BenchMonitors
        from alloy.bully.targets import get_target
        from alloy.bully.types import BullyLimits

        tg = get_target(target)
        self.limits = BullyLimits()
        self.monitors = BenchMonitors(model, self.limits, tg.non_target_bodies, robot_bodies=tg.robot_bodies,
                                      thin_parts=tg.thin_parts)
        if touch not in ("pre", "union"):
            raise ValueError(f"touch must be 'pre' or 'union', got {touch!r}")
        self.collision_model = _collision_model(model, collision_model)
        self.touch, self._pairs = touch, mujoco_shape_pairs(self.collision_model, target)
        self._peak = wp.zeros(3, dtype=wp.int32, device=model.device)
        self.resize(contacts_per_world)
        if touch == "union":
            self._wrap_contacts()

    def resize(self, contacts_per_world: int, triangle_pairs_per_world: int = TRIANGLE_PAIRS_PER_WORLD) -> None:
        """(Re)build the pipeline with pooled budgets per world (all worlds share one buffer of each)."""
        from alloy._vendor import newton

        m = self.collision_model
        self.per_world = int(contacts_per_world)
        self.triangles_per_world = int(triangle_pairs_per_world)
        self.pipeline = newton.CollisionPipeline(
            m, shape_pairs_filtered=self._pairs, rigid_contact_max=self.per_world * m.world_count,
            max_triangle_pairs=max(_MIN_TRIANGLE_PAIRS, self.triangles_per_world * m.world_count))
        self.contacts = self.pipeline.contacts()
        self.post = self.pipeline.contacts() if self.touch == "union" else None
        self._counters = _narrow_counters(self.pipeline.narrow_phase, m.device)
        self._peak.zero_()

    def contact_stats(self) -> dict:
        """Budgets per world, and the peaks Newton asked for since the last reset (pooled rows, pooled triangle
        pairs, contact-reducer losses)."""
        rows, tris, lost = (int(x) for x in self._peak.numpy())
        return {"per_world": self.per_world, "triangle_pairs_per_world": self.triangles_per_world,
                "worlds": self.monitors.model.world_count, "peak_rows": rows, "peak_triangle_pairs": tris,
                "reducer_losses": lost, "max_triangle_pairs": int(self.pipeline.narrow_phase.max_triangle_pairs)}

    def needed(self) -> tuple[int, int] | None:
        """None if no pass overflowed, else the (rows, triangle pairs) per world to score again with: the next power
        of two above 1.25x the peak per world, triangle pairs doubled on reducer losses. A full row buffer counts as
        an overflow: Newton's convex (GJK/MPR) path stops counting once the buffer is full (the mesh paths count past
        it), so with mesh geoms as hulls a peak at capacity can hide dropped rows."""
        st = self.contact_stats()
        w = st["worlds"]
        rows_ok = st["peak_rows"] < st["per_world"] * w
        tris_ok = st["peak_triangle_pairs"] <= st["max_triangle_pairs"] and not st["reducer_losses"]
        if rows_ok and tris_ok:
            return None
        grow = lambda peak, have: max(have, int(2 ** np.ceil(np.log2(max(1.25 * peak / w, 1.0)))))  # noqa: E731
        tris = grow(st["peak_triangle_pairs"], st["triangle_pairs_per_world"])
        if not tris_ok and tris == st["triangle_pairs_per_world"]:
            tris *= 2
        return grow(st["peak_rows"], st["per_world"]), tris

    def _wrap_contacts(self) -> None:
        """After BenchMonitors sets touch flags from the pre-step rows, OR in the post-step rows' touches (same
        kernel, int atomic max; BenchMonitors internals, this bench's only use of them)."""
        import warp as wp

        from alloy.bully import kernels as K

        mon, m, inner = self.monitors, self.monitors.model, self.monitors._contacts

        def contacts_union(cur, contacts) -> None:
            inner(cur, contacts)
            c = self.post
            if c.rigid_contact_max and m.body_count:
                wp.launch(K.contact_touch_kernel, dim=c.rigid_contact_max, device=m.device, inputs=[
                    c.rigid_contact_count, c.rigid_contact_shape0, c.rigid_contact_shape1, m.shape_body, mon._robot,
                    m.body_world, mon._touch_any, mon._touch_robot, mon._world_robot_touch])

        mon._contacts = contacts_union

    def reset(self, state) -> None:
        self.monitors.reset(state)
        self._peak.zero_()

    def _collide(self, state, contacts) -> None:
        self.pipeline.collide(state, contacts)
        wp.launch(_buffer_peak_kernel, dim=1, device=self._peak.device,
                  inputs=[contacts.rigid_contact_count, *self._counters, self._peak])

    def step(self, prev, cur, dt: float) -> None:
        self._collide(prev, self.contacts)
        if self.post is not None:
            self._collide(cur, self.post)
        self.monitors.update(prev, cur, None, None, dt, self.contacts)

    def observer(self, state_in, state_out, control, contacts, dt) -> None:
        self.step(state_in, state_out, dt)

    def report(self) -> dict[str, np.ndarray]:
        return self.monitors.report()


class Replay:
    """The target's model + ``PipelineMonitors`` fed from exported frames, ``worlds`` worlds."""

    def __init__(self, worlds: int, device: str, touch: str = "pre", target: str = DEFAULT_TARGET,
                 contacts_per_world: int = CONTACTS_PER_WORLD) -> None:
        self.model = target_model(target, worlds, device)
        self.pm = PipelineMonitors(self.model, touch, target, contacts_per_world)
        self.limits = self.pm.limits
        self.prev, self.cur = self.model.state(), self.model.state()
        from alloy_t12 import body_names  # noqa: PLC0415 - same naming rule as the Alloy exporter

        self.names = body_names(self.model)

    def _set(self, state, q: np.ndarray, qd: np.ndarray) -> None:
        state.body_q.assign(np.ascontiguousarray(q.reshape(-1, 7), dtype=np.float32))
        state.body_qd.assign(np.ascontiguousarray(qd.reshape(-1, 6), dtype=np.float32))

    def run(self, q: np.ndarray, qd: np.ndarray, t: np.ndarray) -> dict[str, np.ndarray]:
        """``q`` [T, W, B, 7], ``qd`` [T, W, B, 6] in Newton body order; per-world monitor maxima. A pass whose
        contacts overflowed the pooled buffer is run again with a larger one."""
        while True:
            out = self._pass(q, qd, t)
            grow = self.pm.needed()
            if grow is None:
                return out
            log.warning("scorer contact buffers overflowed (%s); again at %s per world", self.pm.contact_stats(), grow)
            self.pm.resize(*grow)

    def _pass(self, q: np.ndarray, qd: np.ndarray, t: np.ndarray) -> dict[str, np.ndarray]:
        self._set(self.prev, q[0], qd[0])
        self.pm.reset(self.prev)
        for f in range(1, q.shape[0]):
            self._set(self.cur, q[f], qd[f])
            self.pm.step(self.prev, self.cur, float(t[f] - t[f - 1]))
            self.prev, self.cur = self.cur, self.prev
        return self.pm.report()


def verdict(extra: dict[str, np.ndarray], limits) -> dict:
    """Per-world severity and S1 cause via Alloy's classifier (no health flags)."""
    from alloy.bully.severity import classify, world_cause
    from alloy.precision.types import HealthReport

    w = next(iter(extra.values())).shape[0]
    health = HealthReport.ok(w)
    sev = classify(health, extra, limits)
    causes = {int(k): world_cause(health, extra, limits, int(k))[0] for k in np.nonzero(sev == 1)[0]}
    return {"severity": sev.tolist(), "s1_worlds": sorted(causes), "s1_causes": causes,
            "monitors": {k: np.asarray(v).tolist() for k, v in extra.items()}}


def score(path: Path, device: str, touch: str = "pre") -> dict:
    z = load(path)
    meta = json.loads(str(z["meta"]))
    q, qd, t = z["body_q"], z["body_qd"], z["t"].astype(np.float64)
    rep = Replay(q.shape[1], device, touch, meta.get("alloy_target", DEFAULT_TARGET))
    idx = _order([str(n) for n in z["body_names"]], rep.names)
    out = verdict(rep.run(q[:, :, idx], qd[:, :, idx], t), rep.limits)
    out.update({"file": str(path), "engine": meta.get("engine"), "test": meta.get("test"), "seed": meta.get("seed"),
                "worlds": int(q.shape[1]), "frames": int(q.shape[0]), "frame_dt": float(np.median(np.diff(t)))})
    if "alloy_s1/severity" in z:   # Alloy's own run: its online verdict beside the replay
        out["online_severity"] = z["alloy_s1/severity"].tolist()
        out["online_monitors"] = {k.split("/", 1)[1]: z[k].tolist() for k in z if k.startswith("alloy_s1/")
                                  and k not in ("alloy_s1/severity", "alloy_s1/first_tick")}
    return out


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("npz", type=Path)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--touch", choices=("pre", "union"), default="pre")
    p.add_argument("--out", type=Path)
    a = p.parse_args()
    res = score(a.npz, a.device, a.touch)
    (a.out or a.npz.with_suffix(".s1.json")).write_text(json.dumps(res, indent=1))
    log.info("%s: S1 worlds %s causes %s", a.npz.name, res["s1_worlds"], res["s1_causes"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
