
---

## The deliverable: a vectorized environment behind one contract

Build, under `~/task`, an environment for the request at the end of this file. You are judged by the harness's own
driver, `~/task/drive.py` (numpy only; do not edit it, it is restored before the scored run), which imports your
`env.py`, steps it through fixed phases and records everything it reads. The scored result is **only** what a fresh
run of `bash ~/task/run.sh` produces.

Deliver in `~/task`:

- `env.py` with `make_env(num_envs: int, seed: int)` returning an env object.
- `run.sh`: activates whatever environment you set up (virtualenv, modules) and runs `python drive.py` from `~/task`
  (it writes `out/rollout.npz` and `out/result.json`). It must work from a clean `out/` directory in the time the
  re-run cap allows, with no network access other than the package index you already used.
- any assets your environment needs (model files, meshes), inside `~/task`.

### What `env.py` provides

- `make_env(num_envs, seed)` -> an object with attributes `num_envs`, `obs_dim`, `act_dim`, `dt` (the control period
  in seconds, one `step` call), and `episode_steps` (an episode lasts this many control steps unless it ends sooner).
- `env.reset()` -> `obs`, shape `(N, obs_dim)`. Resets every world.
- `env.step(action)` -> `(obs, reward, done)` with shapes `(N, obs_dim)`, `(N,)`, `(N,)`. `action` has shape
  `(N, act_dim)`, float32, in [-1, 1], with the meaning and scale the request states. A world whose episode ends
  (`done` true: success, failure as the request defines it, or `episode_steps` reached) resets itself inside that
  same `step` call; the returned `obs` of that world is then its NEW episode's first observation, while `reward` and
  `done` describe the step that ended the old one.
- `env.probe()` -> dict with exactly the probe keys the request lists, each an array `(N, ...)` of ground truth read
  from the simulation: SI units (m, kg, s, rad, N), world frame, z up, quaternions in wxyz order. A scalar quantity
  has shape `(N,)`, not `(N, 1)`. A probe with a variable count (V, P or K entries) keeps one fixed count per run.
  Entries not yet emitted (liquid particles before they exist) may be NaN; nothing else may be.
- `env.success()` -> `(N,)` bool: whether each world is in the request's success state right now (for a world that
  just reset inside `step`, it describes the episode that ended).
- Arrays may be numpy, torch or jax (any device); the driver converts. Actions are handed to `step` in the array type
  of the observation that `reset()` returned.
- Robot-task requests only: `scripted_policy(env, obs) -> action` in `env.py` (module level), which may read
  anything it likes from `env` and should succeed in at least 80 % of worlds within one episode.
- Worlds must differ: any randomness the request names (start positions, targets, friction, ...) is drawn per world
  from `seed`. Two `make_env(N, seed)` calls with the same seed give the same start states (within 1e-3); a
  different seed gives different ones. Later steps are not compared, so GPU non-determinism in contact is fine.

### What the driver does with it

With `N = 16` worlds and seed 0: (1) make, reset, record `obs` and `probe()`; (2) phase `rest`: zero actions for a
few seconds; (3) phases of the harness's own action script (`script.py`: constant, ramp or feedback laws on your
probe values), so the physical response is checked without trusting your policy; (4) phase `random`: seeded random
actions for two episodes; (5) robot tasks: phase `policy` runs your `scripted_policy` for one episode; (6) a second
env with seed 1 (start states must differ) and a third with seed 0 (must match the first); (7) timing of 200 steps at
16 and at 256 worlds (reported, never gated). The checker reads the recorded probe data and judges the physics the
request states: rest states, analytic responses, conservation (cloth stretch, liquid volume, no penetration), and
that the reward is higher in success states than at the start. Do not tune your environment to the driver; build
what the request says and make `probe()` report the truth.

### Self-test

Run `python drive.py --quick` from `~/task` (about a minute; writes `out_quick/`, never `out/`). It exercises every
phase with short durations and prints `PROBLEMS:` with the contract violations it found, or `No problems found by
the driver.` Run it after each change. Run the full `python drive.py` (or `bash run.sh`) at least once before you
finish, and read `out/result.json` (`completed`, `errors`, `violations`, `seeds`).

### If the engine cannot do it

If after real effort the engine you were told to use cannot represent what the request needs, do not substitute
another engine and do not fake the physics. Write `out/result.json` containing
`{"supported": false, "reason": "<what is missing and where you looked in the documentation>"}` and make `run.sh`
produce exactly that file. A decline is judged against what the engine's public documentation says; an environment
that runs with wrong physics is worse than an honest decline.

### Rules

- Use only the engine named above for the physics (other packages only for numpy-style glue). The re-run records
  which modules were imported.
- `run.sh` is what is scored: start from nothing but `~/task`. Do not rely on state left in your shell or on files
  outside `~/task`.
- Stop when `python drive.py` finishes with no problems and you have read the physics in your own `rollout.npz`
  (does the rest state look right? does the response match what you expect?), or when the time runs out.

