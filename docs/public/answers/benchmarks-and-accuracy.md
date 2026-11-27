# What are Alloy's measured throughput, rendering and accuracy numbers?

The Alloy physics engine (`pip install alloy-engine`) publishes throughput rates, batched depth-camera rates, ten
selected closed-form physics checks, thousand-world stability and policy-training success rates in its paper
(arXiv:2611.16312). Every number below comes from the measurements behind that paper, on one NVIDIA L4 or a CPU,
and the snippet runs a deterministic free-fall check on the CPU.

```python
import alloy

scene = alloy.Scene(worlds=4, dt=0.01)
scene.add_ground()
scene.add_box("block", size=(0.05, 0.05, 0.05), pos=(0, 0, 0.5), mass=0.2)
sim = alloy.build(scene, seed=0, device="cpu")
sim.run(1.0)
print(sim.worlds, sim.state(["block"])["block.pos"][:, 2].round(3), sim.check()["ok"])
```

```text
4 [0.025 0.025 0.025 0.025] True
```

All four worlds rest at z = 0.025 m (half the 5 cm box height) and pass their health checks.

## Policy training by coding agents (one NVIDIA L4, `alloy-engine 0.1.0rc2`)

Each agent read only the package's own docs, on a fresh machine. One training run per row, evaluated on held-out
seeds (times exclude first-run GPU kernel compilation):

| Task | Method and worlds | Training time | Held-out success |
|---|---|---|---|
| Unitree Go2 walking (flat ground) | PPO, 4,096 worlds | 4.1 min / 15.5 min | 244/256 (95 %) / 497/512 (97 %) |
| Go2 push recovery (150 N, 0.1 s) | Same 15.5 min policy | 15.5 min | 252/256 and 248/256 (97-98 %) |
| Cable lift (tester-written task) | PPO, 1,024 worlds | ~15 min / ~30 min | 242/256 (94.5 %) / 245/256 (95.7 %) |
| `Pour-v0` (>= 75 % water in bowl) | PPO, 512 worlds | 38.5 min | 118/128 (92 %) |
| Agent-written cup pour (>= 75 % in bowl, 2nd reward version) | PPO, 256 worlds | ~70 min | 256/256 (100 %) |

## Parallel throughput, depth cameras and stability (one NVIDIA L4, warm)

- **Physics throughput (four of nine calibration scenes, 10 ms period, pre-release build, sim-s/s summed over
  worlds):** Franka Panda 1.9 / 6.9 / 74 at 1 / 4 / 64 worlds; three objects dropped in a bin 0.83 / 3.3 / 48 / 313
  at 1 / 4 / 64 / 512 worlds; 200 ml pour 0.53 / 2.0 / 7.1 at 1 / 4 / 64; 0.6 m cotton sheet 0.20 / 0.73 / 4.1 at
  1 / 4 / 64.
- **Batched depth rendering (1,024 worlds, Franka scene, rendering only, two L4 runs):** 62,785-70,786 depth images/s
  at 64x64, 14,372-16,111 at 128x128 and 4,106-4,133 at 256x256. Adding segmentation changed cost by under 1 %.
- **Thousand-world stability:** 1,024 worlds (copies of one ten-block 5 cm tower scene, 0.5 mm starting jitter, 5 s
  simulated) had 0 of 1,024 errors (any block moved or sank more than 1 cm, or a non-finite value) at each of
  three profiles (`precision`, `standard`, `fast`) in two runs (pre-release builds).

## Ten selected closed-form laws (CPU, pre-release build, one run each)

Four laws marked `*` ran at the law's own step size rather than the default step:

| Check | Closed-form reference | Measured error | Alloy's tolerance |
|---|---|---|---|
| Pendulum period `*` | Small-angle period | 8.0e-6 | 1e-3 |
| Momentum | Conservation of momentum | 3.9e-6 | 1e-3 |
| Sliding with friction | Coulomb friction law | 2.8e-3 | 1e-2 |
| Rope around a post `*` | Capstan equation | exact | 2e-2 |
| Rod bending (tip load) | Cantilever deflection | 6.5e-7 | 1e-2 |
| Rod twisting | Torsion formula | 1.1e-6 | 1e-2 |
| Hanging cable `*` | Catenary sag | 2.0e-3 | 1e-2 |
| Column under load `*` | Euler buckling load | 1.3e-2 | 5e-2 |
| Liquid at rest | Hydrostatic wall force | 1.1e-2 | 3e-2 |
| Soft solid under strain | Neo-Hookean stress | 2.8e-3 | 5e-2 |

## Keep in mind

- First-run GPU kernel compilation takes minutes and is cached afterwards; warm rates exclude compilation.
- GPU runs with contact do not repeat bit for bit; CPU runs do
  ([details](../../known_gaps/determinism-gpu-rigid.md)).
- Full list: [Keep in mind](../keep-in-mind.md).

## Next

- Estimate wall time and memory before you build: `alloy.estimate(scene, worlds, seconds)`.
- Parallel training: [GPU-parallel engines for RL](gpu-parallel-engines-for-rl.md).
- World health and blow-ups: [Did my simulation blow up?](did-my-simulation-blow-up.md),
  [World health](../guides/health.md).
- Domain randomisation and sysid:
  [Sim-to-real and system identification](sim-to-real-and-system-identification.md).
