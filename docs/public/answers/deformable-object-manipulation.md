# Which simulators handle deformable object manipulation: cloth, rope and soft bodies?

The Alloy physics engine (`pip install alloy-engine`) simulates cloth, cables and soft bodies in one scene beside
rigid bodies and robots, and each kind goes to the solver that suits it. This page drapes a sheet on a table and
hangs a clamped cable beside it, prints the solver each object got, and reads the health check.

```python
import alloy

scene = alloy.Scene(dt=0.01)
scene.add_ground()
scene.add_box("table", size=(0.4, 0.4, 0.3), pos=(0, 0, 0.15), static=True)
scene.add_cloth("sheet", size=(0.2, 0.2), pos=(0, 0, 0.4), resolution=8, mass=0.03, self_contact=False)
scene.add_rod("cable", length=0.3, radius=0.003, pos=(0.0, 0.3, 0.5), direction=(1, 0, 0), segments=12,
              clamp="start", route="chain")
sim = alloy.build(scene)
for name, info in sim.routes().items():
    print(name, info["route"])
```

```text
ground static
table static
sheet cloth_vbd
cable chain
```

`sim.routes()` says which solver took each object. The sheet steps in Newton's VBD cloth solver and the cable on the
fast chain of rigid segments. Now run it and read both back:

```python
sim.run(0.3)
state = sim.state(world=0)
sheet, cable = state["sheet.points"], state["cable.points"]
print(sheet.shape, cable.shape)
print(sheet[:, 2].min().round(2), cable[-1, 2].round(2))
print(sim.check()["ok"])
```

```text
(81, 3) (13, 3)
0.3 0.23
True
```

The sheet has 81 vertices and the cable 13 points. The lowest sheet vertex rests at z = 0.3 m, the table top, and the
free end of the cable hangs at 0.23 m, below its clamp at 0.5 m. `sim.check()["ok"]` is True.

A soft body, such as a block of `silicone_40a`, takes a third route, `soft_vbd`; see
[soft-body manipulation in Python](soft-body-manipulation-python.md).

One measured number: a policy that lifts a cable off a table reached 94.5 % (242 of 256 held-out episodes) after about
15 minutes of training in 1,024 worlds, with no world INVALID; alloy-engine 0.1.0rc2, one training run, one NVIDIA L4.
The task was written by the tester, not shipped.
The checkpoint and its evaluation script are on the
[Hugging Face Hub](https://huggingface.co/argonrobotics-ai/alloy-cable-lift).

## Keep in mind

- A clamped cable beside a robot on the CPU steps slowly unless it uses the chain route
  ([details](../../known_gaps/rod-vbd-beside-robot-cost.md)).
- No robot grasp that carries a cotton sheet is verified yet ([details](../../known_gaps/cloth-robot-grasp-carry.md)).
- A cloth falling onto another cloth can pass through it
  ([details](../../known_gaps/cloth-falling-sheets-pass-through.md)).
- Full list: [Keep in mind](../keep-in-mind.md).

## Next

- Soft bodies pressed by a tool: [Soft-body manipulation in Python](soft-body-manipulation-python.md)
- Cloth, liquid and robots together: [Robots, cloth and liquids in one scene](robots-cloth-liquids-one-scene.md)
- Cables in detail: [Cables](../guides/cables.md).
