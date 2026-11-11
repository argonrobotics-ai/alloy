# alloy.check: is this scene physically sound?

`alloy.check(scene)` builds the scene, looks for problems and returns a `CheckReport`. Every finding says what is
wrong, with numbers, and what to change. Pose fixes also come as data you can apply.

```python
import alloy.check

report = alloy.check("library/scenes/demo_peg_d12.yaml")   # or a library scene name, YAML text, a SceneCard
print(report.ok)              # False if any finding is an error
print(report.summary())       # one line per error and warning, with its fix
data = report.to_dict()       # same as json.loads(report.to_json())
```

Command line (only JSON on stdout; exit code 0 = no errors, 1 = errors, 2 = bad input):

```bash
python -m alloy.check library/scenes/demo_peg_d12.yaml --json
python -m alloy.check my_scene.yaml --quick            # load checks only: cards and overlap, a few seconds
python -m alloy.check my_drop.yaml --moving            # the scene moves by design: falls are info, not errors
cat scene.json | python -m alloy.check - --json        # scene as JSON (or YAML) on stdin
```

## What it checks

| Check | What it catches | Error when | Warning when |
|---|---|---|---|
| `cards` | card values out of range, unknown objects or keys, missing notes; the scene does not assemble | any card error | card warnings |
| `overlap` | shapes interpenetrating at the start, in every world (randomized poses included) | deeper than 1 mm | deeper than 0.1 mm |
| `rest_drift` | objects that drop, fall, tip, slide or keep moving with no input | fell > 5 cm, tipped > 45 deg, drifts > 1 cm in the second half | settled > 2 mm, tipped > 10 deg, drifts > 0.5 mm, still moving > 5 cm/s |
| `penetration` | the deepest contact during the run | deeper than 10 mm | deeper than 1 mm |
| `health` | NaN or infinite state, contact capacity overflow, launched bodies, joints pulling apart | any world failed | (info: solver iteration cap) |
| `energy` | passive objects gaining energy with no cause over the run | gain > 50 % of the energy scale | gain > 5 % |

Defaults: 64 worlds for 2 s on a GPU; 2 worlds for 0.5 s on CPU. Pass `worlds=` and `seconds=` to change them,
`run=False` for the load checks only.

The run has a wall budget (`budget_s=25`, CLI `--budget 25`): a slow scene simulates fewer seconds, and an info finding
says how many. `budget_s=None` (CLI `--budget 0`) runs them all. A scene whose CUDA graph capture fails (a part reads
the host inside a step) runs again without graphs: an info finding and `metrics["run"]["graph_error"]` say so, and the
log's traceback names the part.

Pair depths (overlap, and which pair a penetration is) come from the check's own contact probe, a collision pass
beside the simulation. When its buffers overflow (many worlds of scanned meshes: 64 worlds of a mug and three ice
cubes needed 3.3 million triangle pairs for 2.7 million slots), it is rebuilt with room for what the overflow
counted, up to three times; the collision library prints its own overflow line each time. If contacts are still
lost, a warning finding says the pair depths are incomplete. The simulation's contacts are not affected either way.

On a GPU the run uses at most the worlds that fit in free device memory (`alloy.check.memory`): a per-world estimate
from the assembled scene (rigid terms plus each solver's particles, MPM about 40 kB per particle, measured on an L4)
against 80 % of the free memory, at least 1 world. An 80 mm MPM dough ball needs about 0.7 GB per world, so its drop
test runs 26 of the 64 worlds on an empty L4. `settings["worlds"]` is the count that ran, `settings["worlds_asked"]`
the count asked, `settings["worlds_why"]` the reason, and a smaller count is an info finding. A simulation that still
runs out of memory is a `health` error that says to pass fewer worlds.

`penetration` is the deepest contact at any moment of the run; `sim.health()` reports the depth at its latest check,
that is now. The two differ when an object lands: a 0.1 kg box sinks 1.3 mm landing at 1 m/s and 3.9 mm at 2.3 m/s,
then rests at 0.003 mm (CPU). The finding gives both numbers (`metrics["penetration"]["end_pen_m"]` is the end value,
the one `sim.health()` reports after the same run); a peak that is gone at rest says to start the object on its
support or stiffen the contact. A smaller `Scene(dt)` barely changes it (1.30 mm at 10 ms, 1.27 mm at 2 ms).

Measured on one L4 GPU, 64 worlds x 2 s, kernels compiled (the first check on a new machine compiles Warp's kernels,
one to four minutes, and the cache keeps them):

| Scene | Wall time | Simulated |
|---|---|---|
| cube on a table | 3.0 s | 2 s |
| robot arm (`task_emb_e2_panda`) | 6.1 s | 2 s |
| liquid pour (`demo_pour`) | 8.1 s | 2 s |
| peg in a 0.2 mm bore (`demo_peg_d12`) | 11.6 s | 2 s |
| nut on a bolt (`demo_nut_on_bolt`) | 26.9 s | 1.2 s (budget) |
| cloth drop (`demo_cloth_drop`) | 43 s | 0.08 s (budget; pass `worlds=4` for more) |

## One object (importers)

```python
report = alloy.check.check_object("library/objects/mug_stoneware.yaml")   # or a name, or an ObjectCard
```

The object is placed in its authored orientation 0.5 mm above a generated plywood floor and checked like a scene.
Tipping over is a warning here (the finding carries the settled pose).

## A running simulation

```python
out = alloy.check.check_sim(sim)     # {"ok", "problems": [{"world", "what", "fix", ...}], "source", "report"}
```

Overlaps now, health so far and the deepest contact now; rest drift and energy need `alloy.check(scene)`.

## A finding

```json
{"check": "overlap", "level": "error",
 "message": "cube_beech_30 and table_top_plywood overlap by 3 mm at start in every world",
 "fix": "lift cube_beech_30 by 3.5 mm (pose z 0.012 -> 0.0155 m)",
 "objects": ["cube_beech_30", "table_top_plywood"], "value": 0.003, "limit": 0.001, "unit": "m",
 "suggestion": {"object": "cube_beech_30", "field": "objects.1.pose", "pose": [0, 0, 0.0155, 0, 0, 0, 1]}}
```

## Fix loop (importers, scene builders)

```python
card = ...                                        # an alloy.cards.types.SceneCard
report = alloy.check.check_load(card)             # cards + overlap, no simulation
card, applied = alloy.check.apply_fixes(card, report)
report = alloy.check(card)                        # full check with a short run
```

`apply_fixes` applies only mechanical edits (new poses, a close library name for an unknown object). Check again
after applying: one fix can reveal the next problem. See `examples/check/check_and_fix.py`.

## Extra run checks (other packages)

A package can add checks that read the same run instead of simulating again:

```python
from alloy.check.report import Finding

def soft(res, built, index):          # res: alloy.check.run.RunResult, index: alloy.check.objects.ObjectIndex
    kept = ...                        # e.g. from res.samples[-1].particle_q and index.particle_obj
    found = [Finding("soft", "warn", f"the fill kept {kept:.0%} in its cup", "lower the fill level")] if kept < 0.95 else []
    return found, {"kept": kept}

report = alloy.check.check(scene, extra=[soft])     # report.checks["soft"], report.metrics["soft"]
```

The check is named by `fn.check_name`, else `fn.__name__` (lowercase letters, digits and `_`). A function that
raises becomes a warning under its name; the rest of the report stands. Stable for callers:
`alloy.check.run.run_scene`, `RunOptions`, `RunResult`, `Sample` and `alloy.check.objects.ObjectIndex.build`.
