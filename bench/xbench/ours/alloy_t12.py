# SPDX-License-Identifier: Apache-2.0
"""Alloy beside, T12 / embodiments: run Alloy's SB-7 target on B0 episodes, write the P3-2 trajectory npz (Alloy venv).

    PYTHONPATH=. python bench/xbench/ours/alloy_t12.py --seed 100000 --worlds 16 --device cuda:0 --out a.npz

The episodes are rows 0..worlds-1 of ``b0.applied_traces(space, 2048, seed)``, so they are the same (seed, k) episodes
an engine plays. The run is Alloy's own ``EpisodeRunner`` on the pack's target (T12: ``task_t12``, the production
route, WP7's SB-7
setup), so the npz also carries Alloy's own per-world S1 verdict (``alloy_s1/*``) for the scorer to agree with.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(ROOT), str(HERE)]

import b0  # noqa: E402
from packs_io import T12, load_pack, runner as make_runner, target_of  # noqa: E402

log = logging.getLogger("xbench.ours")
PACK = HERE / "packs" / "t12" / "t12.json"   # T12's pack file (other packs: --pack)


class Recorder:
    """Step observer: body_q / body_qd (host copies) every ``every`` solver steps, frame 0 = the reset state."""

    def __init__(self, every: int) -> None:
        self.every, self.steps = max(1, int(every)), 0
        self.q: list[np.ndarray] = []
        self.qd: list[np.ndarray] = []

    def start(self, state, worlds: int) -> None:
        self.steps, self.q, self.qd, self.worlds = 0, [], [], worlds
        self._take(state)

    def _take(self, state) -> None:
        # copy: on CPU ``numpy()`` is a view of the state buffer, which the solver overwrites two steps later
        self.q.append(state.body_q.numpy().reshape(self.worlds, -1, 7).copy())
        self.qd.append(state.body_qd.numpy().reshape(self.worlds, -1, 6).copy())

    def __call__(self, state_in, state_out, control, contacts, dt) -> None:
        self.steps += 1
        if self.steps % self.every == 0:
            self._take(state_out)


def body_names(model) -> list[str]:
    """Pack body names (MJCF names) in Newton body order for one world."""
    labels = list(model.body_label)[: model.body_count // model.world_count]
    return [str(lbl).rsplit("/", 1)[-1] for lbl in labels]


def versions() -> dict[str, str]:
    import mujoco
    from alloy._vendor import mujoco_warp
    from alloy._vendor import newton
    import warp as wp

    from alloy.xbench.adapter import _git

    return {"alloy": str(_git().get("alloy_commit")), "newton": newton.__version__, "warp": wp.__version__,
            "mujoco": mujoco.__version__, "mujoco_warp": getattr(mujoco_warp, "__version__", "?")}


def run(seed: int, worlds: int, device: str, every: int, horizon_s: float = b0.HORIZON_S,
        pack_dir: Path = T12) -> dict[str, np.ndarray]:
    """One batch of ``worlds`` episodes of ``seed`` on the pack's Alloy target; returns the npz arrays."""
    pack = load_pack(pack_dir)
    target = target_of(pack)
    traces = b0.applied_traces(pack["action_space"], b0.SEED_BATCH, seed, horizon_s)[:worlds]
    runner = make_runner(target, worlds, device)
    rec = Recorder(every)
    runner.sim.add_step_observer(rec)
    rec.start(runner.sim.state, worlds)
    t0 = time.perf_counter()
    res = runner.run(traces)
    wall = time.perf_counter() - t0
    dt = float(runner.sim.dt)
    meta = {"engine": "alloy", "engine_version": versions()["alloy"], "versions": versions(), "test": f"{target}_b0",
            "alloy_target": target, "pack": Path(pack_dir).name,
            "dt": dt, "substeps": int(runner.sim.substeps), "record_every": rec.every, "seed": seed, "seeds": [seed],
            "worlds": worlds,
            "row": "recommended", "settings": {"route": f"get_target({target!r}): Alloy route",
                                               "ALLOY_MJW_ROUTE": os.environ.get("ALLOY_MJW_ROUTE", "on (default)")},
            "wall_s": wall, "notes": "Alloy's EpisodeRunner (WP7 SB-7 setup); body_qd is Newton's convention"}
    out = {"meta": np.array(json.dumps(meta)), "body_names": np.array(body_names(runner.model)),
           "t": np.arange(len(rec.q)) * dt * rec.every, "body_q": np.stack(rec.q).astype(np.float32),
           "body_qd": np.stack(rec.qd).astype(np.float32), "alloy_s1/severity": res.severity,
           "alloy_s1/first_tick": res.first_step}
    out.update({f"alloy_s1/{k}": np.asarray(v) for k, v in res.monitors.items()})
    return out


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=b0.FIRST_SEED)
    p.add_argument("--worlds", type=int, default=16)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--record-every", type=int, default=1, help="solver steps per frame (32 = one per 20 ms tick)")
    p.add_argument("--pack", type=Path, default=T12)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    arrays = run(a.seed, a.worlds, a.device, a.record_every, pack_dir=a.pack)
    np.savez_compressed(a.out, **arrays)
    sev = arrays["alloy_s1/severity"]
    log.info("wrote %s: %d frames, %d worlds, S1 worlds %s", a.out, arrays["t"].size, a.worlds,
             np.nonzero(sev == 1)[0].tolist())
    return 0


if __name__ == "__main__":
    sys.exit(main())
