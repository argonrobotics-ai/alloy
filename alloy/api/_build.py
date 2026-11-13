# SPDX-License-Identifier: Apache-2.0
"""``alloy.build(scene, worlds, device, seed)`` (module ``alloy.api._build``): assemble a scene on a device and return an ``alloy.Sim``.

Example::

    import alloy
    scene = alloy.load("demo_peg_d12")               # a library scene, a .yaml card, an MJCF/URDF file, or a Scene
    sim = alloy.build(scene, worlds=16, device="cuda:0", seed=1)
    sim.step(50)
"""

from __future__ import annotations

import logging
import os
import tempfile
import time
from pathlib import Path
from collections.abc import Sequence
from typing import Any, NamedTuple

import numpy as np

from alloy.api import _cpu_note, _robot_random, _robot_random_push, _robots, catalog, fixtures as _fixtures
from alloy.api.errors import BuildError, SceneError, world_limits, wrapped
from alloy.api.scene import Scene
from alloy.coupling import world_clock_api as second
from alloy.route import weld_stiff

log = logging.getLogger("alloy.api")
PROFILES = ("standard", "precision", "fast", "debug")


def quiet_warp() -> None:
    """Silence Warp's start-up banner and per-kernel-module load lines (they flood logs and agent contexts) unless
    ``ALLOY_VERBOSE=1``. Example: ``quiet_warp()`` before the first Warp call."""
    if os.environ.get("ALLOY_VERBOSE", "") not in ("", "0"):
        return
    import warp as wp  # noqa: PLC0415

    wp.config.quiet = True
    if hasattr(wp, "LOG_WARNING"):
        wp.config.log_level = wp.LOG_WARNING


def pick_device(device: str | None = None) -> str:
    """"cuda:0" when there is a GPU and ``device`` is None; "cuda"/"gpu" mean "cuda:0". Example: ``pick_device()``."""
    with wrapped("device selection"):
        quiet_warp()
        import warp as wp  # noqa: PLC0415

        gpus = wp.get_cuda_device_count()
    if device is None:
        return "cuda:0" if gpus else "cpu"
    dev = {"gpu": "cuda:0", "cuda": "cuda:0"}.get(str(device), str(device))
    if dev.startswith("cuda") and not gpus:
        raise BuildError(f"device {device!r} asked for but no CUDA GPU is visible", "pass device='cpu' (slower) "
                         "or run on a machine with an NVIDIA GPU and driver", field="device")
    if dev != "cpu" and not dev.startswith("cuda:"):
        raise BuildError(f"unknown device {device!r}", "pass device='cpu', 'cuda' or 'cuda:<n>'", field="device")
    return dev


def build(scene: Scene | str | Path | dict, worlds: int | None = None, device: str | None = None, seed: int = 0,
          *, profile: str = "standard", contacts: str = "auto", plugins: Sequence[Any] = (),
          processes: int | str | None = None, mpm_sweeps: str = "full", rerun_worlds: int = 0, clock_worlds: int = 0,
          mpm_coupling: str | None = None, contact_capacity: int | None = None,
          constraint_rows: int | None = None, welds: str | None = None) -> Any:
    """Build ``scene`` (a ``Scene``, a path / library scene name for ``alloy.load``, or ``Scene.to_dict()`` data).

    - ``worlds``: parallel copies (default: the scene's ``worlds``);
    - ``device``: "cuda:0", "cpu", or None for the GPU when there is one;
    - ``seed``: seeds randomized values (``Scene.randomize``); same seed, same worlds;
    - ``profile``: "standard" (default), "precision" (sub-mm fits; raises if a world goes INVALID), "fast"
      (throughput, lighter checks) or "debug" (no CUDA graphs);
    - ``contacts``: "auto" (default), "mujoco" or "newton". MuJoCo's own collision treats every mesh as its convex
      hull, so "auto" picks Newton's contact pipeline when the scene has a mesh with collision "sdf" or "mesh"
      (cups, bins, holes) and MuJoCo's otherwise;
    - ``plugins``: your own ``alloy.plugins`` objects (forces, sensors), or functions of the built scene that
      return one; they run inside this simulation (``sim.plugins`` reads them). Heat set on the scene
      (``Scene.set_temperature`` / ``add_heater``) is added for you.
    - ``processes``: on the CPU, split the worlds over this many worker processes (an int, or "auto": one per
      core, at most one per world). Each process steps its own block of worlds (Warp's CPU kernels use one core),
      so many worlds run up to N-cores times faster; a world's trajectory is the same in any process (seeds and
      randomized values go by global world id). It returns an ``alloy.Sim``-like object
      (``alloy.parallel.cpu_worlds.SplitSim``) with the same calls; ``sim.close()`` stops the workers. Default None:
      one process.
    - ``mpm_sweeps``: "full" (default) or "partial": pastes (dough, mayonnaise, ...) and other MPM materials re-sweep
      only grid blocks still moving, same stop test (L4, 256 dough worlds: 1.29x faster, within 0.01 mm; 1 process).
    - ``mpm_coupling``: how a robot or a body coupled to sand, soil or paste (MPM) trades forces with it.
      "standard": two passes per exchange, 1.25 ms exchanges. "auto": one pass at the standard 1.25 ms exchange when the
      scene has a servo-driven robot and every body that starts near the material is that robot's link or welded;
      else standard. It is decided at build, so a free body that reaches the material later (a dropped block) also
      runs one pass, resting within 0.5 mm of standard. "fast": one pass and 2.5 ms exchanges for every body (a
      robot in sand runs faster still; a free body landing on the material rests about 3 mm deeper, one resting on
      it bobs about 15 µm before it settles). None (default): "auto". ``sim.routes()`` names the
      setting and why on the material. One process only (``processes=`` refuses it; ``ALLOY_MPM_COUPLING``, a
      run-wide override, reaches the workers).
    - ``rerun_worlds``: opt-in, 0 (default) is off. In a scene with several solvers (a robot or free body beside
      cloth or a soft body), the engine sometimes re-runs a control period for a few worlds (a contact's force
      jumped, or a world needed a finer exchange). With ``rerun_worlds=n`` those worlds re-run alone in a second
      simulation of n worlds, built with this one (memory for n more worlds), instead of every world re-running.
      The other worlds keep their first run. Results match a whole re-run to rounding (as another world count
      does). Take about worlds // 16. It helps on the GPU: on the CPU's default MuJoCo route, beside liquid or MPM,
      or when more worlds re-run than it holds, a re-run runs as before. ``sim.report()["fine_reruns"]`` counts the
      re-runs it took. Not with ``processes`` or user ``plugins``.
    - ``clock_worlds``: opt-in, 0 (default) is off. In a scene with several solvers, all worlds share one exchange
      step, and it is the smallest any world needs (the world with the lightest object in contact, or the fastest
      cloth). With ``clock_worlds=n``, the up to n worlds that need the smallest step move into a second
      simulation of n worlds, built with this one (memory for n more worlds), and step there at their own step;
      the remaining worlds then share the larger step that the neediest of them asks for. A world moves in and
      out as its need changes. State, recordings and commands read and write the same names and world numbers
      as without it. ``set_pose``, ``settle`` and a held ``apply_wrench`` first move the worlds they touch back
      into the main simulation. Worlds that stay in the main simulation step bit for bit as if the moved ones
      were parked by hand; a moved world matches its run without the option only to rounding (as a different
      world count does). Take about worlds // 16. Raises ``BuildError`` with ``rerun_worlds`` or ``processes``.
      In a scene that cannot take it (liquid, MPM, the contact gate, the CPU's default MuJoCo route, or user
      plug-ins) the build warns, says why, and steps every world at the shared step as before.
      ``sim.report()["world_clock"]`` has the books: ``exchanges`` taken, ``exchanges_shared`` (an estimate of
      what one shared step would have taken, not a run; ``exchanges_shared / exchanges`` is the saving in
      exchanges, not in wall time), ``residents`` (the worlds in the second simulation now) and ``paused`` (why it
      is off this period, or None). On 64 worlds of a robot beside a towel with randomized pad masses (one L4) it
      ran 1.12-1.24x faster with clock_worlds=4; on the CPU, a 4-world rig ran 0.88x (slower): the second
      simulation's fixed cost. ``alloy.envs.VectorEnv`` and ``alloy.live`` refuse such a simulation. No law yet
      covers a moved world that fails, or GPU runs (the speed row above is a measurement, not a law). Limits:
      ``sim.report()["world_clock"]["limits"]`` and docs/guide/worlds.md.
    - ``contact_capacity``: MuJoCo's contact buffer, ``n`` contacts per world on average (``nconmax``: MuJoCo Warp
      keeps one pool of n x worlds that the worlds share). Unless ``constraint_rows`` is given it also sizes the rows
      per world from the model: each contact's rows (3 on the elliptic cone) x 2 n plus the scene's joint-limit,
      equality and friction-loss rows at their maximum (``alloy.api._capacity``); a world that takes more than 2 n
      of the pool runs out of rows first: ROW_OVERFLOW, same fix. None (default) sizes it from the scene: 8 per
      collision shape, at least 64, 6 rows per contact.
      Past the pool, MuJoCo Warp (the GPU's route) drops the last contacts found; the CPU's MuJoCo C route keeps them
      all in its physics, but health reads only this many, so the worlds are flagged the same way. Either way they
      are CONTACT_OVERFLOW and frozen, and the console says so once, at the first overflow (any step), with the n to
      pass; ``sim.health()`` names the worlds. One process only for now (``processes=`` refuses it).
    - ``constraint_rows``: MuJoCo's constraint row buffer, rows per world (``njmax``), whatever the contact buffer
      is. None (default) keeps the sizing above; without ``contact_capacity`` that is 6 rows per contact, which for a
      robot of hundreds of collision shapes is thousands of rows where it uses tens. MuJoCo Warp (the GPU's route)
      runs its constraint and solver kernels over the whole buffer, so rows sized to the scene step faster, with less
      memory: on one L4 at 2,048 worlds a Panda holding 13 rows of 4,080 stepped 1.55x faster with
      ``contact_capacity=64, constraint_rows=384``, a LEAP hand 2.2x, a G1 (512 worlds, 256 and 768) 1.7x, with the
      same contacts, rows and solver iterations. A contact takes 3 rows (6 with torsional and rolling friction),
      and each joint limit in use, friction-loss joint and equality constraint its own: leave about twice what a
      world uses. With too few, the worlds are ROW_OVERFLOW and frozen (``sim.health()``), never stepped on missing
      rows without a flag. The CPU's MuJoCo C route sizes its own rows: the option changes nothing there. One
      process only for now.
    - ``welds``: "stiff" (the default: MJCF weld / connect rows at MuJoCo's default solref / solimp take the
      attachments' stiff rule for the rigid step, ``docs/known_gaps/weld-stretch-under-load.md``) or "mujoco".

    The first build of a new scene compiles GPU kernels (about a minute; cached afterwards).
    Example: ``sim = alloy.build(scene, worlds=1024)``."""
    scene = _as_scene(scene)
    _fixtures.check_riders(scene.items, scene.extra.get("attachments") or ())   # a kinematic fixture is one body
    n = _world_count(scene, worlds, profile)
    dev = pick_device(device)
    native = _contacts(scene, contacts)
    tau = _mpm_tau(mpm_sweeps)
    pair_clock = _pair_clock(mpm_coupling)
    plugins = _checked_plugins(plugins)
    second.check_slots(rerun_worlds, n)
    _robot_random.refuse_rerun_worlds(scene, rerun_worlds)
    _robot_random_push.refuse(scene, rerun_worlds, processes)
    _robots.sysid_refuse_rerun_worlds(scene, rerun_worlds)
    second.check_slots(clock_worlds, n, "clock_worlds")
    second.refuse_with(clock_worlds, rerun_worlds, processes)
    cap = _capacity(contact_capacity, constraint_rows)
    weld_stiff.check(welds, processes, BuildError)
    if processes is not None:                          # the default path stays the one-process build, no extra import
        split = _split_build(scene, n, dev, seed, processes, profile=profile, contacts=contacts, plugins=plugins,
                             tau=tau, rerun_worlds=rerun_worlds, pair_clock=pair_clock, capacity=cap)
        if split is not None:
            return split
    from alloy.integration.mpm_clock import requested  # noqa: PLC0415

    with requested(pair_clock), weld_stiff.requested(welds), world_limits(n):   # read when the stepper / model is wired
        sim = _mark_mpm(build_block(scene, n, dev, seed, profile=(profile, native), plugins=plugins, capacity=cap),
                        tau)
        make = _pack_factory(scene, sim.device, seed, (profile, native), cap)     # gap #10b's second simulations
        if rerun_worlds:
            second.rerun_pack(sim, make, int(rerun_worlds), bool(plugins))
        if clock_worlds:
            second.clock_pack(sim, make, int(clock_worlds), bool(plugins))
    return _cpu_note.after_build(sim, device)        # GEO P6: one line on CPU speed when no GPU was found


def _world_count(scene: Scene, worlds: int | None, profile: str) -> int:
    """The checked profile and world count of ``build`` (BuildError naming the field otherwise)."""
    if profile not in PROFILES:
        raise BuildError(f"unknown profile {profile!r}", f"pass one of {PROFILES}", field="profile")
    n = scene.worlds if worlds is None else worlds
    if not isinstance(n, (int, np.integer)) or n < 1:
        raise BuildError(f"worlds must be an int >= 1, got {worlds!r}", "pass worlds=1 (or more)", field="worlds")
    return int(n)


def _checked_plugins(plugins: Sequence[Any]) -> list[Any]:
    """``build``'s plug-ins, checked (``alloy.api._plugins.check``) when there are any."""
    if not plugins:
        return list(plugins)
    from alloy.api._plugins import check  # noqa: PLC0415

    return list(check(plugins))


def _split_build(scene: Scene, n: int, dev: str, seed: int, processes: int | str, *, profile: str, contacts: str,
                 plugins: list[Any], tau: float | None, rerun_worlds: int, pair_clock: str | None = None,
                 capacity: Capacity | None = None) -> Any:
    """``build(..., processes=)``: the worlds split over worker processes, or None when that is one process."""
    from alloy.parallel.cpu_worlds import build_processes, process_count  # noqa: PLC0415

    count = process_count(processes, n, dev)
    if count <= 1:
        return None
    if rerun_worlds:
        raise BuildError("rerun_worlds works in one process for now", "drop processes= or rerun_worlds=",
                         field="rerun_worlds")
    _one_process(tau)
    if pair_clock is not None:
        raise BuildError("mpm_coupling is not passed to worker processes yet", "build in one process (processes=None "
                         "or 1), or set ALLOY_MPM_COUPLING for the run", field="mpm_coupling")
    if capacity is not None:
        field = "contact_capacity" if capacity.contacts is not None else "constraint_rows"
        raise BuildError(f"{field} is not passed to worker processes yet", "build in one process "
                         f"(processes=None or 1), or leave {field}=None", field=field)
    return build_processes(scene, n, seed, count, profile=profile, contacts=contacts, plugins=plugins)


class Capacity(NamedTuple):
    """MuJoCo buffer sizes per world asked for in ``build``: contacts (nconmax) and constraint rows (njmax); None
    keeps the scene's own."""

    contacts: int | None
    rows: int | None


def _capacity(contact_capacity: Any, constraint_rows: Any) -> Capacity | None:
    """``contact_capacity`` and ``constraint_rows`` checked: each None or an int >= 1 (BuildError naming the field
    otherwise). None when neither is set."""
    sizes = []
    for field, value, unit, example in (("contact_capacity", contact_capacity, "contacts per world", 512),
                                        ("constraint_rows", constraint_rows, "constraint rows per world", 384)):
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1):
            raise BuildError(f"{field} must be an int >= 1 ({unit}), got {value!r}",
                             f"pass e.g. {field}={example}, or None to size it from the scene", field=field)
        sizes.append(None if value is None else int(value))
    return None if sizes == [None, None] else Capacity(*sizes)


def _sized(built: Any, capacity: Capacity | None) -> None:
    """``built``'s MuJoCo entries hold ``capacity.contacts`` contacts per world (nconmax), with 6 constraint rows per
    contact or the entry's own row budget, whichever is larger (a small capacity must not starve a robot's limit and
    equality rows). ``capacity.rows`` sets the rows per world (njmax) itself: the user sized them."""
    if capacity is None:
        return
    import dataclasses  # noqa: PLC0415

    from alloy.scene.entries import ROWS_PER_CONTACT  # noqa: PLC0415

    def sized(e: Any) -> Any:
        kw = dict(e.solver_kwargs)
        if capacity.contacts is not None:
            kw["njmax"] = max(ROWS_PER_CONTACT * capacity.contacts, int(kw.get("njmax") or 0))
            kw["nconmax"] = capacity.contacts
        if capacity.rows is not None:
            kw["njmax"] = capacity.rows
        log.info("contact capacity %s per world (njmax %s) on MuJoCo entry %s", kw.get("nconmax"), kw.get("njmax"),
                 e.name)
        return dataclasses.replace(e, solver_kwargs=kw)

    built.entries = [sized(e) if e.capability == "mujoco" else e for e in built.entries]


def _mpm_tau(mpm_sweeps: str) -> float | None:
    """``mpm_sweeps``'s partial-sweep threshold (None: every sweep full)."""
    from alloy.integration import mpm_solve  # noqa: PLC0415

    return mpm_solve.sweeps_tau(mpm_sweeps, BuildError)


def _pair_clock(mpm_coupling: str | None) -> str | None:
    """``mpm_coupling`` checked (None: the engine's default, "auto"); BuildError naming the field
    otherwise."""
    if mpm_coupling is None:
        return None
    from alloy.integration.mpm_clock import CLOCKS  # noqa: PLC0415

    if str(mpm_coupling) not in CLOCKS:
        raise BuildError(f"unknown mpm_coupling {mpm_coupling!r}", f"pass one of {CLOCKS} or None (the default)",
                         field="mpm_coupling")
    return str(mpm_coupling)


def _one_process(tau: float | None) -> None:
    """"partial" sweeps build in one process: a split into worker processes (count > 1) does not pass them yet."""
    if tau is not None:
        raise BuildError("mpm_sweeps='partial' is not passed to worker processes yet", "build in one process "
                         "(processes=None or 1) or keep mpm_sweeps='full'", field="mpm_sweeps")


def _mark_mpm(sim: Any, tau: float | None) -> Any:
    """``sim`` with its MPM solves set to sweep partially at ``tau`` (alloy.integration.mpm_worlds reads it)."""
    if tau is None:
        return sim
    from alloy._vendor.newton.solvers import SolverImplicitMPM  # noqa: PLC0415
    from alloy.integration.mpm_solve import MODEL_TAU, attach_notes, route_note  # noqa: PLC0415
    from alloy.integration.stepper import entry_solvers  # noqa: PLC0415

    setattr(sim.built.model, MODEL_TAU, tau)
    stepper = getattr(sim.engine, "stepper", None)
    for solver in (entry_solvers(stepper) if stepper is not None else {}).values():
        if isinstance(solver, SolverImplicitMPM):       # sim.routes() names it (WP12's review), beside any other note
            attach_notes(solver, "sweeps", route_note(sim.built.model))
    return sim


def build_block(scene: Scene, worlds: int, device: str, seed: int, *, profile: tuple[str, str | None],
                plugins: list[Any], block: tuple[int, int] | None = None, capacity: Capacity | None = None) -> Any:
    """``build`` after its checks: ``Sim`` of ``scene`` with ``worlds`` worlds, or of only global worlds
    ``block`` = (offset, count) of them (a worker of ``build(..., processes=N)``; global world ids key every random
    draw). Example: ``build_block(scene, 8, "cpu", 0, profile=("standard", None), plugins=[], block=(4, 4))``."""
    from alloy.api import _robot_random_apply  # noqa: PLC0415
    from alloy.api.sim import Sim  # noqa: PLC0415

    workdir = tempfile.mkdtemp(prefix="alloy-scene-")
    t0 = time.perf_counter()
    engine = _assemble_and_wire(scene, worlds, device, seed, profile, Path(workdir), plugins, block, capacity)
    log.info("built %d world(s) on %s in %.1f s (profile %s)", worlds if block is None else block[1], device,
             time.perf_counter() - t0, profile[0])
    _robot_random_apply.attach(engine)                 # when="reset" rows redraw in engine.reset (none: no-op)
    _robots.sysid_attach(scene, engine)                 # add_robot(sysid=): the fitted action delay
    sim = Sim(engine, device=device, profile=profile[0], workdir=workdir, scene=scene)
    _robot_random_push.attach(sim, scene, seed)        # scene.add_push rows (none: no-op)
    return sim


def _pack_factory(scene: Scene, device: str, seed: int, profile: tuple[str, str | None],
                  capacity: Capacity | None) -> Any:
    """n -> a ``Sim`` of n worlds of ``scene`` (no user plug-ins): ``rerun_worlds``' and ``clock_worlds``' packs."""
    return lambda n: build_block(_with_worlds(scene, n, trim=True), n, device, seed, profile=profile, plugins=[],
                                 capacity=capacity)


def _as_scene(scene: Any) -> Scene:
    if isinstance(scene, Scene):
        return scene
    if isinstance(scene, dict):
        return Scene.from_dict(scene)
    if isinstance(scene, (str, Path)):
        from alloy.api._load import load_scene  # noqa: PLC0415

        return load_scene(scene)
    raise BuildError(f"cannot build a {type(scene).__name__}", "pass an alloy.Scene, a scene file path or a "
                     "library scene name", field="scene")


def _contacts(scene: Scene, contacts: str) -> str | None:
    """The ``native_contacts`` override for the profile: None keeps the profile's choice."""
    if contacts not in ("auto", "mujoco", "newton"):
        raise BuildError(f"unknown contacts {contacts!r}", "pass contacts='auto', 'mujoco' or 'newton'",
                         field="contacts")
    if contacts != "auto":
        return contacts
    concave = [n for n, it in scene.items.items()
               if it["kind"] == "mesh" and it["shape"]["mesh"]["collision"] in ("sdf", "mesh")]
    if concave:
        log.info("contacts: Newton's pipeline (non-convex meshes %s; MuJoCo would collide their convex hulls)",
                 concave)
        return "newton"
    return None


def _assemble_and_wire(scene: Scene, worlds: int, device: str, seed: int, profile: tuple[str, str | None],
                       workdir: Path, plugins: list[Any], block: tuple[int, int] | None = None,
                       capacity: Capacity | None = None) -> Any:
    from alloy.api import (_anchor, _contact_relax, _heat, _joint_yield, _liquid_accuracy, _pour,  # noqa: PLC0415
                           _quasi_static, _rigid_step, _robot_random_apply, _rods, _worlds)
    from alloy.api._capacity import rows_from_contacts  # noqa: PLC0415
    from alloy.api._materialize import materialize  # noqa: PLC0415
    from alloy.api._plugins import build_engine  # noqa: PLC0415

    heat = _heat.plugin_for(scene)
    yielding, qs = _joint_yield.plugin_for(scene), _quasi_static.plugin_for(scene)
    sized = _pour.at_spacing(_with_worlds(scene, worlds), _liquid_accuracy.spacing(scene.liquid_accuracy))
    mat = materialize(sized, workdir)
    with wrapped("build"):
        import warp as wp  # noqa: PLC0415

        from alloy.cards.library import CardLibrary  # noqa: PLC0415
        from alloy.cards.roots import assets_root  # noqa: PLC0415
        from alloy.scene.assembler import SceneAssembler  # noqa: PLC0415

        lib = CardLibrary(roots=catalog.card_roots([workdir, *scene.roots]), assets=assets_root())
        fetch_imported(lib, mat.cards)
        with wp.ScopedDevice(device):
            offset, count = block or (0, worlds)
            step = _rigid_step.request(scene, profile[0], [*plugins, heat, yielding, qs], mat.per_world_pos,
                                       announce=offset == 0,      # its warning once, as the liquid's below
                                       newton_contacts=profile[1] == "newton")
            router = _rods.with_clamps(_liquid_accuracy.router_for(scene), scene.items)   # add_rod(route='chain')
            router = _quasi_static.with_scales(_fixtures.with_bodies(router, scene.items), scene)  # fixture: 1 body
            built = SceneAssembler(lib, router, rigid_step=step).assemble(
                lib.scene(mat.scene_file.stem), device, world_count=count, world_id_offset=offset)
            if offset == 0:                                 # once: one process, or the first worker of a split
                _liquid_accuracy.warn(scene, built)
            _reseed(built, seed)
            _sized(built, capacity)
            built.kinematic_fixtures = _fixtures.kinematic_groups(scene.items)  # NameIndex: '<name>', parts
            _worlds.place(built, mat.scene, _fixtures.own_rows(mat.per_world_pos, scene.items), seed, block)
            _rods.clamp_ends(built, scene.items)            # add_rod(clamp=...): end segments kinematic
            _robots.sysid_install(scene, built)              # add_robot(sysid=): before the randomize draws
            _robot_random_apply.install(built, scene, seed)   # Scene.randomize robot / gravity rows (none: no-op)
            _quasi_static.apply(built, scene)               # set_quasi_static_mass_scale: refusals, masses x s
            weld_stiff.apply(built)                         # build(welds="stiff"): default MJCF weld rows
            if mat.pours:
                from alloy.api.names import NameIndex  # noqa: PLC0415

                _pour.make_spares(built, NameIndex.from_built(built), mat.pours)
            name, native = profile
            extra = {"native_contacts": native} if native else {"contacts_auto": True}  # may pick Newton's contacts
            auto_rows = capacity.contacts if capacity is not None and capacity.rows is None else None
            with rows_from_contacts(auto_rows):             # contact_capacity alone: njmax from the model at put_data
                engine = build_engine(built, name, seed=seed, plugins=plugins + [p for p in (heat, yielding, qs) if p],
                                      **extra)
            engine._alloy_heated = tuple(scene.thermal["objects"])
            _quasi_static.register(engine)                  # its shares to the health monitor (QUASI_STATIC)
            engine._alloy_anchors = _anchor.anchors(mat.scene)   # objects placed by an anchor: .pos reads it
            _contact_relax.install(engine, built, scene)       # scene.relax_contact_depth: health marks
            if mat.pours:
                _pour.start(engine, mat.pours)
            return engine


def fetch_imported(lib: Any, cards: list[str]) -> None:
    """Fetch the files (meshes) of every placed card that comes from an imported library and lacks them, from
    ``$ALLOY_ASSET_STORE``; a clear error when there is no store. On a machine with no imported library and no store,
    a card ``lib`` lacks is fetched from the object library on the Hub (``alloy.assets.hub.lazy``) and ``lib`` is
    rescanned. Example: ``fetch_imported(lib, ["ycb_025_mug"])``."""
    for name in dict.fromkeys(cards):
        if not lib.has_object(name):
            _from_hub(lib, name)
            continue
        root = catalog.import_root_of(lib.path_of("objects", name))
        if root is None:
            continue
        from alloy.compose.fetch import STORE_ENV, fetch_object, missing  # noqa: PLC0415

        if not missing(root, name):
            continue
        if not os.environ.get(STORE_ENV):
            from alloy.assets.hub.lazy import fetch_fix  # noqa: PLC0415

            raise SceneError(f"{name!r} (imported library {root.name}) has no mesh files on this machine",
                             f"{fetch_fix(name)}, set ${STORE_ENV} to the asset store (gs://..., https://... or a "
                             f"folder) so Alloy fetches them, or run python -m alloy.importers.{root.name} fetch",
                             field=name)
        log.info("fetching %s from the asset store (imported library %s)", name, root.name)
        fetch_object(root, name, None)


def _from_hub(lib: Any, name: str) -> None:
    """Card ``name`` from the object library on the Hub, when this machine has no imported library and no store
    (else the assembler's own error names it). ``SceneError`` with the Hub's fix when that fails."""
    from alloy.assets.hub.lazy import ensure, fetch_fix, hub_mode  # noqa: PLC0415
    from alloy.assets.hub.source import HubError  # noqa: PLC0415

    if not hub_mode():
        return
    try:
        ensure(lib, name)
    except HubError as exc:
        raise SceneError(f"object card {name!r}: {exc.message}", fetch_fix(name, exc), field=name) from exc


def _reseed(built: Any, seed: int) -> None:
    """Redraw the scene's randomized values at ``seed`` (assembly draws at 0); ``reset`` then redraws at it too."""
    plan = getattr(built, "_plan", None)
    if not seed or plan is None or not any(p.randomize for p in plan.scene.objects):
        return
    from alloy.scene.assembler import digest_model  # noqa: PLC0415
    from alloy.scene.randomize import apply_randomization  # noqa: PLC0415

    apply_randomization(built, int(seed))
    built.model_digest = digest_model(built.model)


def _with_worlds(scene: Scene, worlds: int, *, trim: bool = False) -> Scene:
    """The scene with ``worlds`` worlds; ``trim`` keeps the first rows of per-world positions (estimates)."""
    if worlds == scene.worlds:
        return scene
    out = Scene.from_dict(scene.to_dict())
    out.worlds = worlds
    for name, item in out.items.items():
        rows = item.get("per_world_pos")
        if rows is None or len(rows) == worlds:
            continue
        if not trim or len(rows) < worlds:
            raise BuildError(f"{name!r} has {len(rows)} per-world positions but worlds={worlds}",
                             f"pass worlds={len(rows)} or give {worlds} positions", field="worlds")
        item["per_world_pos"] = rows[:worlds] if worlds > 1 else None
    return out
