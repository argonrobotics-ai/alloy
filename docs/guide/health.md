# Health, checks and errors

## Health

The engine watches every world while it steps. `sim.health()` returns JSON-ready data:

```
{"ok": bool, "monitored": bool, "invalid_worlds": [...], "degraded_worlds": [...],
 "worlds": [{"severity": "OK" | "DEGRADED" | "INVALID", "flags": [...], "seen_flags": [...], "max_penetration_m": float,
             "peak_penetration_m": float,
             "problem": str, "fixes": [str, ...], "first_bad_step": int}, ...],
 "retired_particles": [int, ...] | None, "refined_periods": [int, ...] | None}
```

`max_penetration_m` is the deepest contact at the latest check; `peak_penetration_m` is the deepest since the build
or the last `sim.reset()` / `sim.settle()`.

| Flag | Meaning | What to change |
|---|---|---|
| NONFINITE_Q / _QD / _PARTICLE | a position, velocity or particle became NaN or inf | lower `dt`; realistic masses and stiffness; no initial overlaps |
| PEN_WARN | objects overlap by more than 1 mm, or by more than half a part thinner than 2 cm (DEGRADED). For rigid bodies, robots and cloth on them the flag shows the latest check and clears when they come apart; `peak_penetration_m` and `seen_flags` keep it for the run | lower `dt`, or `profile="precision"` for tight fits |
| PEN_FAIL | overlap past the limit, 1 cm or half the thinner part, for 0.1 s of simulated time; or past twice the limit (2 cm, through a thin part, or 10 % of a body larger than 20 cm) at four checks in a row. A bounce or a landing that dips past the limit for a few milliseconds reads PEN_WARN; a fast body's one-step dip into a thin part is FAST_CONTACT / TUNNEL. INVALID: the world is frozen at its last good state. What these two flags do not catch yet: `docs/known_gaps/health-mujoco-penetration.md` | start objects apart, lower `dt`, thicken thin parts |
| PAIR_PEN_WARN / PAIR_PEN_FAIL | a precision pair near / past its clearance gate | lower `dt`, slow the insertion |
| SPEED_FAIL | something moved faster than 20 m/s | remove initial overlaps and huge forces; lower gains |
| ENERGY_JUMP | energy rose with nothing adding it; checked only in scenes with no motors, joint targets or kinematic objects. Three bounds, each a share of the scene's energy scale (its weight times its size): (1) more than 1 % in one physics step: DEGRADED (alone it leaves `sim.check()` ok); (2) more than 10 % above the lowest level the scene has held for a quarter second: DEGRADED until `sim.reset()`, and `sim.check()` is not ok (the world's `energy_gain` in `sim.health()` gives the percent; the first quarter second after a start or reset only sets the level, and a world where outside work is counted, such as a held `sim.apply_wrench` or a force plug-in, is judged by the other two bounds only); (3) more than 50 % above the start: INVALID, an explosion | lower `dt` (a contact too stiff for the step), remove initial overlaps; for a mesh object that never comes to rest, try `collision="convex"` or `"decompose"`. A force plug-in's push (wind, a field) and a held `sim.apply_wrench` are counted as work from outside, and `sim.set_pose` restarts the measure, so none of them trips this flag |
| JOINT_FAIL | a joint, or a weld / connect between two bodies (MJCF `<equality>`), is more than 1 mm or 1 mrad apart (DEGRADED). Measured every 10th physics step (every step under `profile="precision"`; also under `"fast"`, since 2026-11-10), and it stays on until a measurement finds the joint closed again. An opening that lasts under 10 steps, or a load that repeats in step with them, can fall between measurements and be missed. A robot's own joints cannot come apart in MuJoCo, so there it means a weld or connect | lower the load, gains or `dt`; a stiffer weld (`solref`) sags less; check joint limits in the robot file |
| CONTACT_OVERFLOW / ROW_OVERFLOW | more contacts or constraint rows than the buffers hold: results not trustworthy. The console says so once, at the first overflow, naming the first flagged world and, when MuJoCo's buffer overflowed, the capacity to pass | when MuJoCo's buffer overflowed (the console line says so and names n), `alloy.build(scene, contact_capacity=n)`: n contacts per world on average (MuJoCo Warp's pool of n x worlds is shared by the worlds; default 8 per collision shape, at least 64); it sizes the constraint rows too (each contact's rows x 2 n plus the scene's joint-limit, equality and friction-loss rows), so a ROW_OVERFLOW (a world holding more than 2 n contacts) has the same fix, or a larger `constraint_rows=m` if you set it; else, or as well, fewer touching objects per world, coarser meshes (`collision="convex"`) |
| UNCONVERGED | the solver did not converge | lower `dt`, or `profile="precision"` |
| START_PEN | objects overlapped at the start | move them apart before the run (`sim.settle` does not undo an overlap); a library object placed with `anchor="center"` or `"origin"` can reach into what is beside it: leave room or use `anchor="base"`. The ground counts: a body that starts deep inside it is START_PEN, named with its depth, and usually ENERGY_JUMP too (the contact throws it out; the world goes INVALID): place objects above their support |
| SPIN | a body turned more than half a radian in one physics step, so contacts can add energy and fling it | make the thin part heavier or thicker, or step rigid bodies finer: `scene.rigid_step = 0.000625`, or `Scene(dt=0.001)` |
| CLOTH_STRETCH | a cloth stretched over 50 % past its rest size (DEGRADED); over 500 % the sheet blew up (INVALID) | lower `dt`, or give the cloth more `mass=` or `thickness=`; keep its mesh edges (size / (resolution - 1)) longer than its thickness: a lower `resolution=`, or `self_contact=False`. A `cloth_accuracy="fast"` drape stretches 12 %, under it: read the shape |
| SELF_CONTACT_OVERFLOW | a cloth or soft body touched itself or another one in more places than its contact buffers hold (they grow with the scene up to 256 contacts per vertex or edge, and to 4 times their starting memory, at least 256 MiB); the extra contacts were dropped, so those surfaces can pass into each other (DEGRADED; stays on for the rest of the run). On request health also counts the triangle pairs that then pass through each other (crossing triangles, or a vertex that went through to the other side): `ALLOY_SELF_CONTACT_CROSS=report` at build shows the count as `self_contact` in the world's health row; `ALLOY_SELF_CONTACT_CROSS=invalid` makes such a world INVALID, named in `sim.check()`. Off by default | stack fewer cloth layers (deep piles of folded sheets are the likely case); `self_contact=False` only for a sheet that never folds and is the scene's only cloth or soft body. Past those limits the build logs the fix once (engine layer: `alloy_self_contact_row_bytes` in the VBD entry's kwargs; docs/known_gaps/vbd-self-contact-buffer-overflow.md) |
| FAST_CONTACT | a free body closed on a static box, sphere, capsule or cylinder by more than 5 mm in one rigid step, the part is thinner along the approach than twice that travel, and the body ended within 5 mm of it: it can pass through the part (DEGRADED while it happens: the flag clears 64 rigid steps after the approach, so read it during the run, not only at the end). Not raised against the ground plane or a thick part. Judged: free rigid bodies against static shapes, not static meshes, kinematic bodies or two moving bodies | keep a body's travel per rigid step under 5 mm: `scene.rigid_step = 0.0003125` (a rigid-only scene) or `Scene(dt=0.0003)`; or thicken the static part; or slow the body ([limit and numbers](../known_gaps/rigid-tunnelling.md)) |
| TUNNEL | a point inside a moving body (its shapes' centres and long-axis points: a free body or a robot link) passed through a static box, plane, sphere, capsule, cylinder or box-like convex mesh: in and out of the opposite side, out of another side from 2 mm deep, across it in one step, or the whole body below it. INVALID, the world is frozen and `sim.check()` names the body and the shape. Not judged: a point that starts inside the shape, two moving bodies, kinematic objects, other static meshes ([limits](../known_gaps/rigid-tunnelling.md)) | step rigid bodies finer (`scene.rigid_step = 0.0003125` in a rigid-only scene, or `Scene(dt=0.0003)`), slow the body, or make the parts thicker; no travel per step is safe for every part (a pin of radius 1.5 mm passed a 5 mm pane at 1.4 mm per step), so measure your own part at your speed and read `sim.check()` |
| STRAIN_RANGE | a soft body was squeezed or stretched over 60 %, past its model's range: motion past it is approximate | press with an edge rather than a flat face, or with less force |
| FAR_FROM_ORIGIN | an object went more than 4 m from the origin: a notice, listed in `flags` while the world stays OK (a walking or mobile robot passes 4 m in seconds); past 16 m it is INVALID under the precision profile | check that it is supported (a ground?) and was not launched |
| RUNG_PINNED | a solver asked for a smaller step than it was pinned to | lower `dt` |
| FALLBACK_USED / DET_DOWNGRADED | notices, listed in `flags` while the world stays OK: the engine used a fallback path / runs on this device do not repeat bit for bit (every GPU run with rigid bodies, robots or cloth, at every profile: `docs/known_gaps/determinism-gpu-rigid.md`) | nothing, unless results look wrong; run on the CPU when you need the same bits |
| LATCHED | the world was frozen at its last good state | fix the first problem listed for that world |

An INVALID world is frozen at its last good state while the other worlds go on, so a failed world can look calm
afterwards. Under `profile="fast"` (checks every 10 steps) that is its last checked state, up to 9 steps before the
failure at one step per period ([worlds.md](worlds.md), The rigid step). Read health (or `sim.check()`) before you
trust or report a result. `traj.meta["health"]` holds the health at the end of each `sim.run`.

`severity` and `flags` are those of the latest check (one after every physics step, several per control period): a
DEGRADED flag goes away when its cause does, so a contact that was 3 mm deep for a few steps is not in `flags` a
second later. `seen_flags` keeps every flag any check raised in that world since the build, the last
`sim.reset()` or the last `sim.settle()` (including checks of a coupled period the engine then re-ran at a finer
step), so read it (in `sim.health()` or `traj.meta["health"]`) to learn what happened during a run.
`sim.check()` lists each flag that came and went as a warning ("PEN_WARN earlier in this run: ..."); it does not
change `ok`. INVALID flags never go away by themselves: they stay in `flags` until the reset.

Two per-world counts, one entry per world:
- `retired_particles`: liquid particles that left the scene. None without a liquid.
- `refined_periods`: control periods that the coupling re-ran one exchange step finer, that period only. A world's
  periods are re-run when its coupling watchdog trips (an energy or residual rule). The count runs from the build, so
  `sim.reset()` does not zero it. With `processes=` the workers' counts are joined in world order. It is None when
  nothing is coupled (a scene of only rigid bodies and robots).
  - A non-zero count is not a problem and raises no flag: the other worlds keep their results.
  - A count that is high, or keeps growing, in one world points at a coupling instability there
    ([coupling-shared-exchange-clock.md](../known_gaps/coupling-shared-exchange-clock.md), Escalations).

## Checks

`sim.check()` returns `{"ok": bool, "problems": [{"world", "what", "fix", ...}], ...}` for the run so far: every
problem with the world it happened in and what to change. Use it as the last line of every script you hand back:
`assert sim.check()["ok"], sim.check()["problems"]`. `ok` is False for an INVALID world (or a run that stopped); a
DEGRADED world passes it, with two exceptions: one whose energy gain passed ENERGY_JUMP's 10 % bound, and one that
started with an overlap of more than 1 cm (START_PEN). A flag that came and went is listed in `problems` with
`level: "warn"`; a DEGRADED flag that is still on is not, so read `sim.health()` (`degraded_worlds`, each world's
`flags` and `seen_flags`) when a DEGRADED flag matters, such as PEN_WARN. Notices (FAR_FROM_ORIGIN, DET_DOWNGRADED,
FALLBACK_USED) are listed in `flags` and never make a world DEGRADED.

## Precision contact

For fits of a millimetre or less (pegs in holes, connectors, threads), plain contact is too soft. Alloy has precision
pairs: exact-shape contact between two named features, with a clearance gate the health monitor enforces.
- Declare them on both object cards: `precision_features` (a named cropped region and its `clearance`) and
  `precision_partners` (`"<other object>:<feature>"`); see `library/objects/peg_d12.yaml` and
  `block_bore_d12_2.yaml`, and `cards.md` for writing your own.
- Place such objects with `scene.add_object(name, card=...)` (or load a library scene such as `demo_peg_d12` or
  `demo_nut_on_bolt`) and build with `profile="precision"`.
- `sim.report()["gap"]` has the measured penetration of each precision pair.
- A box-built hole needs no cards: `scene.add_block_with_hole` makes a precise-fit scene by default (fine rigid step,
  stiff contacts; see [Scenes](scenes.md)).

```python
import alloy

sim = alloy.load("demo_peg_d12", worlds=2, profile="precision")
sim.step(5)
h = sim.health()
print(h["ok"], [w["severity"] for w in h["worlds"]], sim.check()["ok"])
```

## Errors

Every error from the API says what went wrong and what to change: `e.message`, `e.fix`, `e.field` (the argument or
card field), `e.to_dict()` for JSON. Engine errors during a build or step are wrapped into one of these, with the
original as `e.__cause__`.

| Class | When |
|---|---|
| `alloy.SceneError` | a bad argument or card: negative size, unknown material, duplicate name |
| `alloy.BuildError` | the engine could not build the scene: routing, capacity, missing device |
| `alloy.SimulationError` | stepping failed or a world became INVALID under `profile="precision"` |
| `alloy.UnknownNameError` | a body, joint or object name is not in the scene; the message lists the names |
| `alloy.FeatureUnavailableError` | a feature this install does not have |

```python
import alloy

try:
    alloy.Scene().add_box("cube", size=(0.1, 0.1))
except alloy.SceneError as e:
    print(e.to_dict())        # {"error": "SceneError", "message": ..., "fix": ..., "field": "size"}
```
