# Physics check

Two ways to ask "is this physically sound?":

- `sim.check()` on a `Sim` you have run: problems found while it ran (health events, penetration), each with the
  world, what happened and the fix. Last line of every script: `assert sim.check()["ok"], sim.check()["problems"]`.
- `alloy.check(scene_file, worlds=..., seconds=...)` on a scene before you use it: validates the cards, looks for
  overlaps at the start in every world, runs a short simulation and reports rest drift, penetration, health events
  and energy. It never raises for a broken scene; every finding is a sentence that ends in the change to make.

```python
import alloy
import alloy.check

scene = alloy.Scene(worlds=2, dt=0.01)
scene.add_ground()
scene.add_box("table", size=(0.6, 0.4, 0.4), pos=(0, 0, 0.2), static=True)
scene.add_box("block", size=(0.05, 0.05, 0.05), pos=(0, 0, 0.42), mass=0.2)     # 5 mm into the table top
card = scene.save("checked_scene")                     # the scene as card files
report = alloy.check(str(card), worlds=2, seconds=0.5)
print(report.ok, report.checks)                       # {"cards": "pass", "overlap": "fail", ...}
for f in report.findings:
    if f.level == "error":
        print(f.text())                               # "... overlap by 5 mm at start -> lift block by ..."
```

- Scenes where things are meant to fall or move: `alloy.check(..., expect_rest=False)`; otherwise objects that are
  still moving at the end are reported as unsupported.
- `report.to_json()` is the same report as JSON; `alloy.check(..., run=False)` does the load checks only (cards and
  overlap; seconds, no kernels to compile).
- Command line: `python -m alloy.check scene.yaml --json` (exit 0 ok, 1 errors, 2 bad input).
- `alloy check --contacts newton` (MCP: `"contacts": "newton"`) runs the check with that contact pipeline, as
  `alloy build --contacts` does; without it, the check keeps the profile's own pipeline. A check of an open
  simulation (`sim=<id>`) checks it as it is now and refuses check-run arguments (scene, seconds, worlds, moving,
  quick, budget).
- `alloy.check.apply_fixes(card, report)` applies the suggested start poses to a scene card; see
  `examples/check/check_and_fix.py`.
