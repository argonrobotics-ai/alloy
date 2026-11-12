# How do I run a physics simulator on a Mac without a GPU?

The Alloy physics engine runs on the CPU when there is no NVIDIA GPU, with the same code and the same API. Install
it in a Python virtual environment with `pip install alloy-engine`, build your scene as usual, and pass
`device="cpu"` to `alloy.build` if you want to say so explicitly.

```python
import numpy as np
import alloy

scene = alloy.Scene(worlds=4, dt=0.01)
scene.add_ground()
heights = np.array([[0, 0, 0.3], [0, 0, 0.5], [0, 0, 0.7], [0, 0, 0.9]])
scene.add_sphere("ball", 0.03, pos=heights, restitution=0.6)     # one start per world
sim = alloy.build(scene, device="cpu")
sim.run(0.1)
print(sim.state(["ball"])["ball.pos"][:, 2].round(2))
print(sim.check()["ok"])
```

```text
[0.25 0.45 0.65 0.85]
True
```

Four balls start at four heights, one per world. After 0.1 s each has fallen the same 5 cm. `device="cpu"` forces
the CPU; without it Alloy picks `cuda:0` when there is one, else the CPU.

On an Apple-silicon Mac with no NVIDIA GPU, measured in an independent cold test on macOS arm64, the quick start ran
in 44 s including the first compile, and a 9 x 9 vertex towel ran about 20x slower than real time.

For robots, install a C compiler first (`xcode-select --install` on a Mac): rigid bodies and robots then step in
MuJoCo C, and a robot arm runs about real time on one core. Cloth, liquid and soft bodies are much slower on a CPU
and use one core.

## Keep in mind

- Cloth, liquid and soft bodies on a CPU are much slower than real time and use one core; check `sim.routes()`
  before a long run.
- CPU and GPU runs give the same physics model, not the same bits, so a contact-rich outcome such as a grasp can come
  out differently. Do not compare them step for step.
- The page's program was run on Linux, not on a Mac; the Mac figures above come from the separate cold test.
- Full list: [Keep in mind](../keep-in-mind.md).

## Next

- [Quickstart](../quickstart.md)
- [Cloth](../guides/cloth.md)
- [Errors and fixes](../errors.md)
