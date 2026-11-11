# Writing an Alloy plug-in

A plug-in adds physics Alloy does not have: a **force** (magnets, drag, hydraulics, ink deposition), a **sensor**
(accelerometer, range finder), an **object** (a parametric part built from bodies and joints) or a **solver**. The
harness then checks it automatically: its physics laws, crashes, determinism, world independence and speed. This
page is everything you need; the templates and the three built-ins (`alloy/plugins/library/`) are worked examples.

## 1. Five commands

```bash
python -m alloy.plugins new my_wind --kind force   # force | sensor | object | solver; makes ./my_wind/, passes as is
python -m alloy.plugins check my_wind --quick      # crash + laws, ~10-30 s on CPU, JSON report
python -m alloy.plugins check my_wind              # + determinism, world independence, graph replay, speed
python -m alloy.plugins check my_wind --sandbox    # the same, confined (section 8)
python -m alloy.plugins describe magnet            # any plug-in's parameters, as JSON
```

Exit code 0 = every check passed. Each failed check carries `fix`: what to change. The JSON result goes to stdout
and every log line (Warp's kernel messages too) to stderr; if your tool merges the two streams, add
`--out report.json` to get the report as a clean file. Every command also takes a JSON request: `echo '{"command":
"check", "target": "ink", "quick": true}' | python -m alloy.plugins --request -`. From Python: `report =
alloy.plugins.check(MyWind(speed=3.0)); print(report.summary())`.

`new my_wind --dir D` creates the project folder `D/my_wind/` (pyproject.toml, README.md, tests/) holding the
package `D/my_wind/my_wind/` with your code in `plugin.py`. It works from an installed wheel and from a source
checkout alike: the templates are modules of the package (`alloy.plugins.templates`). `check` and `describe` take the project folder
(`my_wind`, from the folder you ran `new` in), the package folder (`my_wind/my_wind`), that `plugin.py`,
`"module:Class"` or a registered name; `--config '{"speed": 3.0}'` sets parameters.

Workflow: scaffold, change the physics in `plugin.py`, write the laws that prove it right, run `--quick` until it
passes, then the full check.

## 2. Anatomy (a complete force plug-in)

```python
from dataclasses import dataclass
import warp as wp
import newton
from alloy.plugins import ForcePlugin, Forces, Law, LawResult, TestScene, body_xform
from alloy.plugins.funcs import add_wrench

@wp.kernel
def _drag_kernel(body_qd: wp.array[wp.spatial_vector], body: wp.array[wp.int32], c: wp.float32,
                 body_f: wp.array[wp.spatial_vector]):
    w = wp.tid()                       # one thread per world
    b = body[w]                        # this world's body (global index)
    qd = body_qd[b]
    f = -c * wp.vec3(qd[0], qd[1], qd[2])
    add_wrench(body_f, b, wp.spatial_vector(f[0], f[1], f[2], 0.0, 0.0, 0.0))

class Drag(ForcePlugin):
    """Linear drag F = -c·v on one body [N·s/m]. Example: Drag(body="box", coefficient=10.0)."""
    name = "drag"                                   # registry name, snake_case, unique
    summary = "linear drag on one body"             # one line for `list`

    @dataclass
    class Config:                                   # parameters; JSON-able types only
        body: str = "box"                           # label of the body   <- comments show in `describe`
        coefficient: float = 10.0                   # N·s/m

    def setup(self, ctx):                           # once: look things up, allocate, register state
        self.body = ctx.bodies(self.config.body)

    def apply(self, state, forces: Forces, dt):     # every step, before the solver: ADD forces
        wp.launch(_drag_kernel, dim=self.ctx.world_count,
                  inputs=[state.body_qd, self.body, float(self.config.coefficient)],
                  outputs=[forces.body_f], device=self.ctx.device)

    def test_scene(self):                           # the scene the harness runs
        return TestScene(build=falling_box, ground=False, periods=50)

    def laws(self):                                 # what must be true
        return [Law("terminal_velocity", check_terminal)]
```

Config field types: `str`, `int`, `float`, `bool`, `None`, and tuples/lists of them (vectors as tuples).
`Drag(coefficient=2.0)`, `Drag({"coefficient": 2.0})` and `Drag(Drag.Config(coefficient=2.0))` are the same.
Unknown fields raise with the valid names. Validate parameters in `Config.__post_init__` (raise `ValueError`).
Two instances in one simulation need different labels: `Drag(label="drag2")`.

## 3. The four kinds and their hooks

| Kind | Base class | You implement | Called |
|---|---|---|---|
| force | `ForcePlugin` | `apply(state, forces, dt)` and/or `update(state_in, state_out, dt)` | every exchange (solver step): `apply` before the solver, `update` after |
| sensor | `SensorPlugin` | `measure(state_in, state_out, dt)` | every exchange, after the solver |
| object | `ObjectPlugin` | `build(builder, xform, label) -> [body ids]` | when a scene is built |
| solver | `SolverPlugin` | `capability() -> SolverCapability` (and a `newton.solvers.SolverBase`) | registered; entries use it by name |

Common to all: `setup(ctx)`, `check_dt(dt)` (raise `PluginError` if unstable at this step, say which knob to
change), `test_scene()`, `laws()`, `read()` (host copies of your state, `{name: [W, ...]}`), `speed_budget` (class
attribute, default 2.0: step time with the plug-in at most 2x without). Force plug-ins: set `internal = True` when
every force has an equal and opposite partner in the scene (the harness then checks zero net force and moment).
Objects: `drop_height` (class attribute) places the default drop test; override `test_scene` for other tests.

`setup(ctx)` gets a `PluginContext`:

| | |
|---|---|
| `ctx.bodies("label")`, `ctx.shapes(...)`, `ctx.joints(...)` | `wp.array` int32 `[W]`: the global index of that item in every world. Errors list the labels that exist |
| `ctx.labels("body")`, `ctx.has("pen")` | what is in world 0 |
| `ctx.zeros(shape, dtype)`, `ctx.full(shape, value, dtype)`, `ctx.array(values, dtype)` | device arrays on the model's device |
| `ctx.gravity` | world 0's gravity vector, numpy `(3,)` (`model.gravity` has one row per world) |
| `ctx.model` | the `newton.Model` (`body_mass`, `body_com`, `body_inertia`, `shape_transform`, `shape_scale`, `gravity`, ...) |
| `ctx.world_count`, `ctx.device`, `ctx.control_dt` | |
| `ctx.time` | `[W]` float64 seconds since each world's reset, advanced on the device (use it in kernels) |
| `ctx.contacts` | Newton contacts of the step, or `None` (MuJoCo runs its own): prefer geometry over contacts |

Register arrays in `setup`:

- `self.x = self.add_state("x", ctx.zeros((ctx.world_count, ...)))`: anything the next step reads (reservoirs,
  volumes, ink maps, sensor readings). First dimension = worlds. Snapshotted, reset, rolled back and digested.
- `self.u = self.add_input("u", ctx.zeros(ctx.world_count))`: a command the user sets between periods with
  `plugin.set_input("u", value_or_per_world_values)`.

## 4. Conventions and helpers

SI units. Newton's state: `body_q[b]` = world transform of the body frame (x, y, z, qx, qy, qz, qw);
`body_qd[b]` = (linear velocity of the centre of mass, angular velocity), world frame; `body_f[b]` = (force at the
centre of mass, torque), world frame; `model.body_com[b]` = centre of mass in the body frame. Particles:
`particle_q/qd [N] vec3`, `forces.particle_f [N] vec3`. Gravity is -z, 9.81 m/s².

Warp helpers (`from alloy.plugins.funcs import ...`, use inside kernels): `body_point(body_q, b, p_local)`,
`body_vector(body_q, b, v_local)`, `com_world(body_q, body_com, b)`, `point_velocity(body_qd, com, b, p)`,
`wrench_at(force, point, com)` (a force applied off the centre of mass, with its torque), `torque_wrench(tau)`,
`add_wrench(body_f, b, w)`. Host helpers for laws (`from alloy.plugins.npgeom import ...`): `quat_rotate(q, v)`,
`transform_point(xf, p)`.

## 5. Rules every hook follows (the harness checks them)

1. **Kernels only.** `apply/update/measure` launch `wp.launch(...)`; no `.numpy()`, no allocation, no Python `if`
   on device values, no Python counters. The step may run inside a captured graph. Python values in `inputs=` are
   frozen at capture, so anything that changes during a run lives in a device array (`add_input`, `add_state`,
   `ctx.time`) and is updated inside kernels. The harness enforces this: hooks that call `.numpy()`,
   `wp.synchronize*` or allocate raise during its runs, and `graph_replay` replays captured periods (CPU graphs too)
   and compares them with an eager run.
2. **Add, never assign** forces: other plug-ins share `forces.body_f`.
3. **Deterministic.** Each output element is written by exactly one thread. No float `wp.atomic_add`, and note that
   `a[i] += x` inside a kernel *is* an atomic: write `a[i] = a[i] + x`. The simplest safe pattern is one thread
   per world (`dim=ctx.world_count`) looping over that world's few items; for big per-world grids use one thread
   per cell that gathers its contributions. No host randomness or wall-clock time; seed `wp.rand_init(seed, w)`.
4. **World isolation.** World `w` only touches world `w`'s items (`self.body[w]`, `state_x[w, ...]`). The harness
   runs identical worlds (they must stay bit-identical) and perturbs worlds 1.. (world 0 must not change).
5. **Stable.** Forces are explicit: a stiffness `k` on mass `m` needs `sqrt(k/m)·dt < 1` and a damping `c` needs
   `c·dt/m < 1`. Check this in `check_dt` and say what to change (see `library/hydraulic.py`).

## 6. Test scenes and laws

```python
def falling_box(b: newton.ModelBuilder):                 # ONE world; the harness replicates it
    box = b.add_body(xform=body_xform(0.0, 0.0, 10.0), label="box")      # a free body
    b.add_shape_box(box, hx=0.05, hy=0.05, hz=0.05, cfg=newton.ModelBuilder.ShapeConfig(density=1000.0))

def check_terminal(run) -> LawResult:
    b = run.bodies("box")[0]                             # world 0's box (global index)
    m = float(run.model.body_mass.numpy()[b])
    got = -float(run.trace.body_qd[-1, b, 2])            # last period, body b, vz
    want = m * 9.81 / run.plugin.config.coefficient      # run.plugin: this run's copy of your plug-in
    return LawResult.within((got - want) / want, 0.01, f"{got:.4g} vs {want:.4g} m/s", "F must be -c·v")
```

`TestScene(build, worlds=3, periods=25, control_dt=0.02, dt=2e-3, solver="mujoco", solver_kwargs={}, ground=True,
ground_mu=None, gravity=(0, 0, -9.81), companions=None, init=None)`. `periods` are control periods (20 ms each);
the solver steps at `control_dt/2^k <= dt` (1.25 ms by default). `init(sim)` runs after the build: set inputs
(`sim.plugins["my_label"].set_input(...)`) or start velocities (`sim.state.joint_qd`, then
`newton.eval_fk(sim.model, sim.state.joint_q, sim.state.joint_qd, sim.state)`; a free body has 6 dofs: vx, vy, vz,
wx, wy, wz). `companions` returns other plug-ins the scene needs.

`Law(name, check, scene=None, doc="", config=None)`: `scene=None` reuses `test_scene()`; `config` is a dict of some
of your Config fields that this law overrides (the run holds a fresh copy of *your* plug-in, as `run.plugin`).
`check(run)` returns `LawResult(passed, value, bound, detail, fix)` or `LawResult.within(value, bound, detail,
fix)`. Rotated start poses: `body_xform(x, y, z, q=(qx, qy, qz, qw))`. With `gravity=(0, 0, 0)` and
`ground=False` a free body stays where it starts.

What `run` gives a law:

| | |
|---|---|
| `run.trace.body_q [K, B, 7]`, `body_qd [K, B, 6]`, `particle_q [K, N, 3]`, `t [K]` | state at t = 0 and after each period (K = periods + 1), all worlds, global indices |
| `run.trace.body_f [K, B, 6]` | the plug-in forces of each period's last step |
| `run.trace.plugin("label", "name")` | `[K, W, ...]`: your registered state over time (`read()` keys) |
| `run.bodies("label")` | `[W]` global body ids; `[0]` is world 0 |
| `run.plugin` | this run's instance of the plug-in under test: `run.plugin.config`, `run.plugin.label`, your helper methods |
| `run.model`, `run.host` | the model; every plug-in in the run (`run.host["label"]`) |
| `run.probe(body_q)` | plug-in forces `[B, 6]` (force xyz, torque xyz per body) at poses `body_q [B, 7]` of every body, with zero velocities, without stepping: `q = run.trace.body_q[0].copy(); q[b, 2] = 0.3; f = run.probe(q)` |

Good laws compare to closed-form physics: a force law (value at two distances), a conservation law (energy, mass,
volume, ink), a steady state (terminal velocity, equilibrium pressure, P·A = m·g), a null case (no effect when
out of range). Bound them by the integration error, not by luck: energy drift shrinks with `dt`.

## 7. What the harness checks

| Check | Fails when | Usual fix |
|---|---|---|
| contract | no name/summary/docstring, hook missing | fill them in |
| float_atomics | a float atomic or `a[i] += x` in a kernel | one writer per element (rule 3) |
| crash | exception (including a hook that read back, synced or allocated), NaN/inf, a body faster than 20 m/s or 100 m away | traceback in `fix`; guard divisions, `check_dt` |
| hook_determinism, hook_identical_worlds, hook_isolated_worlds, hook_graph | your hooks alone, replayed on fixed recorded states: differ run to run, between equal worlds, leak between worlds, or differ when captured as a graph | rules 1, 3 and 4 |
| determinism | two identical runs differ bit-wise | rule 3; no host randomness |
| graph_replay | captured-and-replayed periods differ from eager ones (on a CPU both arms run MuJoCo Warp: the default MuJoCo C route reads the host each step and cannot be captured) | rule 1: no Python state that changes per step |
| identical_worlds / isolated_worlds | worlds differ or leak into each other | rule 4 |
| action_reaction (`internal = True`) | net force or moment of your forces is not zero | equal and opposite forces at the same point |
| mass_inertia, settles (objects) | massless body, unphysical inertia, keeps moving after the drop | density on every shape; joint limit stiffness/damping |
| your laws | `abs(value) > bound` | the `fix` you wrote |
| speed | step time with the plug-in > `speed_budget` x without | fewer launches, fuse loops into one kernel |

`--device cuda:0` runs the same checks on a GPU with real CUDA graphs. There MuJoCo's contacts are not run-to-run
deterministic by default, so a whole-run repeat check (determinism, worlds, graph replay) that fails while all four
hook checks pass is the engine's: it is skipped with that reason (the CPU run, the default, tests it end to end;
its graph replay on MuJoCo Warp, the rest on the scene's own route).
Solver plug-ins get no such pass: there the solver is the plug-in.

## 8. The sandbox

`--sandbox` runs the check in a child process that can read only Python, the installed packages, Alloy and system
folders, can write only its own temporary folder and a sandbox kernel cache (Linux Landlock), has no network (user
and network namespaces; Landlock also denies TCP), runs without credentials in its environment, with CPU, file and
memory limits and a wall-clock timeout. Your plug-in therefore must not read files outside its own folder (pass
data folders with `SandboxPolicy(extra_read=(...))` from Python) or use the network. The sandbox protects the
machine; it does not make the report tamper-proof (plug-in and harness share a process), so review code you did
not write before trusting its pass. The first sandboxed check copies the Warp kernel cache once (~0.5 GB).

## 9. Using plug-ins in a simulation

```python
import alloy.plugins as ap
sim = ap.simulation(built, [ap.create("magnet", bodies=("a", "b"), moments=(1.0, 1.0))])   # alloy.sim.Simulation
sim.plugins["magnet"].set_input(...)     # if it has inputs
for _ in range(50):
    sim.step_period()
print(sim.plugins.read()["magnet"])      # {label: {name: [W, ...]}}
```

For graph capture: `ap.simulation(built, plugins, cls=IntegratedSimulation, use_graph=True)` (from
`alloy.integration.simulation`). `built` is a `BuiltScene` (from Alloy's scene assembler, or `alloy.plugins.harness.build_model` +
`alloy.testing.integration.built_from_model` for a raw Newton model). With your own stepper:
`stepper = ap.PluginHost(built.model, plugins, control_dt=built.control_dt).install(stepper)` before building the
simulation. Objects go into the builder: `HydraulicCylinder(bore=0.03).build(builder, xform, "lift")`. Solvers:
`alloy.core.registry.register_solver(MySolver().capability())`, then an entry with `capability="my_solver"`.

Publish a plug-in package with an entry point and `pip install -e .`; then `ap.available()` and every command find
it by name:

```toml
[project.entry-points."alloy.plugins"]
my_wind = "my_wind.plugin:MyWind"
```

## 10. Pitfalls we hit

- MuJoCo mixes friction by **max**: the ground plane's default μ is 1.0, so a slippery object still sticks. Set
  `TestScene(ground_mu=0.1)`.
- `builder.add_body` makes a free body. For jointed parts use `add_link` + `add_joint_*` + `add_articulation`, and
  give joint limits `limit_ke`/`limit_kd`; Newton's defaults bounce for seconds (the `settles` check catches it).
- MuJoCo contacts are soft: strong forces (magnets) push bodies into each other. Laws should use frames before
  contact.
- A free body dropped onto the ground needs about 0.3 s to settle; read "at rest" values at the end of a long run.
- Worlds are not offset in space: every world uses the same coordinates.
- Trace values are per period (20 ms); a law about a fast event should use a short `control_dt` or `run.probe`.
- Kernel argument types must match: pass `float(x)` for `wp.float32`, `wp.vec3(*v)` for `wp.vec3`.

## 11. Built-in examples

- `magnet` (`library/magnet.py`): point dipoles, pairwise forces and torques, one thread per world; laws: inverse
  fourth power at two distances (probe), energy conservation, action-reaction.
- `ink` (`library/ink.py`): state-only (`update`), a pen tip deposits into a per-world grid when it touches a paper
  shape; laws: ink conserved, nothing when lifted, track length = travel.
- `hydraulic` + `hydraulic_cylinder` (`library/hydraulic.py`): an object (barrel, rod, prismatic joint) and a
  lumped-fluid force with a pump input, relief valve and `check_dt`; laws: volume balance, P·A = m·g, rod speed
  = Q/A, relief caps pressure.

Runnable demos: `examples/plugins/`.
