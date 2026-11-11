# SPDX-License-Identifier: Apache-2.0
"""Newton and MuJoCo Warp are vendored in alloy/_vendor (coordinator ruling 2026-11-07, scr/WP0.md): Alloy runs on
those copies only, whatever newton / mujoco_warp a user has installed, and the copies are the pinned commits."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import warp  # kernel annotations below resolve through this module's globals

ROOT = Path(__file__).resolve().parents[2]
VENDOR = ROOT / "alloy" / "_vendor"


def _decoys(tmp_path: Path) -> Path:
    """Top-level newton / mujoco_warp packages that raise on import: any import of them fails loudly."""
    for name in ("newton", "mujoco_warp"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "__init__.py").write_text(f'raise ImportError("DECOY {name} imported")\n')
    return tmp_path


def _run(code: str, tmp_path: Path) -> str:
    env = dict(os.environ, PYTHONPATH=os.pathsep.join((str(_decoys(tmp_path)), str(ROOT))))
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr[-3000:]
    return out.stdout.strip().splitlines()[-1]


def test_alloy_runs_on_the_vendored_engines_only(tmp_path: Path) -> None:
    """Import, build and step a CPU scene, resolve every PRIVATE_HOOKS entry and the EPA users with decoy top-level
    packages first on sys.path: nothing reaches them (lazy lookups included), and none is in sys.modules."""
    code = """
import sys
import alloy
from alloy.core.newton_compat import missing_private_hooks
from alloy.contact import mjw_epa
import alloy._vendor.mujoco_warp._src.collision_convex, alloy._vendor.mujoco_warp._src.collision_flex
s = alloy.Scene(dt=0.01); s.add_ground(); s.add_box("b", size=(0.1, 0.1, 0.1), pos=(0, 0, 0.3), mass=1.0)
sim = alloy.build(s, device="cpu"); sim.step(5)
top = sorted(m for m in sys.modules if m.split(".")[0] in ("newton", "mujoco_warp"))
print(missing_private_hooks(), sorted(set(mjw_epa.state().values())), top)
"""
    assert _run(code, tmp_path) == "[] ['vendored'] []"


def test_alias_gives_third_party_scripts_the_vendored_modules(tmp_path: Path) -> None:
    code = """
from alloy._vendor import alias_top_level_names, newton as vn, mujoco_warp as vm
alias_top_level_names()
import newton, mujoco_warp
from newton.solvers import SolverXPBD
print(newton is vn, mujoco_warp is vm, SolverXPBD is vn.solvers.SolverXPBD)
"""
    assert _run(code, tmp_path) == "True True True"


def test_versions_and_pins_agree() -> None:
    from alloy._vendor import VERSIONS, mujoco_warp, newton
    from alloy.core.newton_compat import MJWARP_PIN, NEWTON_PIN

    assert (newton.__version__, mujoco_warp.__version__) == (VERSIONS["newton"], VERSIONS["mujoco_warp"])
    text = (VENDOR / "VENDORED.md").read_text()
    for pin in (NEWTON_PIN, MJWARP_PIN):
        assert f"`{pin[:8]}` ({pin})" in text, pin
    script = (ROOT / "scripts" / "vendor_engines.py").read_text()
    assert f'NEWTON_SHA = "{NEWTON_PIN}"' in script and f'MJW_SHA = "{MJWARP_PIN}"' in script


def test_no_alloy_file_imports_the_top_level_engines() -> None:
    """scripts/vendor_engines.py --check: every import of newton / mujoco_warp outside alloy/_vendor is vendored
    (a new `import newton` fails here: write `from alloy._vendor import newton`)."""
    out = subprocess.run([sys.executable, str(ROOT / "scripts" / "vendor_engines.py"), "--check"], cwd=ROOT,
                         capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]


@pytest.mark.parametrize("pkg", ["newton", "mujoco_warp"])
def test_every_vendored_file_keeps_its_licence_and_the_notice(pkg: str) -> None:
    from scripts_vendor import NOTICE  # noqa: PLC0415

    assert any((VENDOR / pkg / name).is_file() for name in ("LICENSE", "LICENSE.md"))
    missing = [str(p.relative_to(VENDOR)) for p in sorted((VENDOR / pkg).rglob("*.py"))
               if NOTICE not in p.read_text().split("\n")]
    assert not missing, missing[:10]


@pytest.fixture(autouse=True, scope="module")
def _script_module():
    """scripts/vendor_engines.py importable as `scripts_vendor` (scripts/ is not a package)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("scripts_vendor", ROOT / "scripts" / "vendor_engines.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["scripts_vendor"] = module
    spec.loader.exec_module(module)
    yield
    sys.modules.pop("scripts_vendor", None)


# ---- SCR-WP0-COEXIST: Alloy beside a user's own newton / mujoco_warp in one process -------------------------------

def test_vendored_engines_use_warp_itself() -> None:
    """Only `module=` kernel calls go through alloy._vendor._warp (func and get_module keep Warp's call and read their
    name through _alloy_warp.module_name); `wp` stays warp (Warp's codegen tells builtins such as wp.tid by `caller is
    warp`, and func finds its overloads in the caller's locals)."""
    import importlib

    for name in ("alloy._vendor.newton._src.geometry.narrow_phase", "alloy._vendor.mujoco_warp._src.smooth",
                 "alloy._vendor.mujoco_warp._src.collision_convex"):
        mod = importlib.import_module(name)
        assert mod.wp is warp, name
        assert mod._alloy_warp.__name__ == "alloy._vendor._warp", name


def test_wrappers_rename_only_unique_keys_and_named_modules() -> None:
    from alloy._vendor import _warp as W

    def plain(a: warp.array(dtype=float)):
        pass

    k = W.kernel(plain)                                          # no module: warp.kernel itself
    assert k.key == warp.kernel(plain).key and not k.key.startswith(W.PREFIX)
    u = W.kernel(enable_backward=False, module="unique")(plain)
    assert u.key == W.PREFIX + plain.__qualname__.replace(".", "__").replace("<", "").replace(">", "")
    n = W.kernel(plain, module="coexist_named_case")
    assert n.module.name == W.PREFIX + "coexist_named_case"
    own = warp.get_module(W.module_name("coexist_module_case"))
    assert W.kernel(plain, module=own).module is own
    with pytest.raises(ValueError, match="not one of Alloy's own"):        # a shared Module object (WP8's review)
        W.kernel(plain, module=warp.get_module("coexist_foreign_case"))
    names = (None, "unique", "alloy._vendor.newton.x_1", W.PREFIX + "y", "z")
    assert [W.module_name(m) for m in names] == [None, "unique", "alloy._vendor.newton.x_1", W.PREFIX + "y",
                                                  W.PREFIX + "z"]


IMPORT = "import warp as wp\n"


@pytest.mark.parametrize("src", [
    "from warp import kernel\n" + IMPORT,
    IMPORT + "k = wp.kernel(*args)(f)\n",
    IMPORT + "g = wp.func(f, **opts)\n",
    IMPORT + "m = wp.get_module(**opts)\n",
    IMPORT + "import warp\nk = warp.kernel(f, **opts)\n",
    IMPORT + "import warp\nm = warp.get_module('x')\n",
    IMPORT + "import warp as w2\nk = w2.kernel(f, module='x')\n",
    IMPORT + "make = wp.kernel\n",
    IMPORT + "import functools\nmake = functools.partial(wp.kernel, module='x')\n",
])
def test_the_rewrite_refuses_forms_it_cannot_see_into(src: str) -> None:
    """Each hides a module name from the rewrite, so --check fails on it (WP8's reviews)."""
    from scripts_vendor import WARP_CALLS  # noqa: PLC0415

    with pytest.raises(WARP_CALLS.CallsError):
        WARP_CALLS.rewrite(src, "case.py")


def test_the_rewrite_is_in_place_and_idempotent() -> None:
    from scripts_vendor import WARP_CALLS  # noqa: PLC0415

    src = ("import warp as wp  # noqa: F401\nimport inspect\nOPTS = inspect.signature(wp.kernel).parameters\n"
           "s = 'é'; m = wp.get_module(\n    'é_mod')\n@wp.kernel(enable_backward=False, **OPTS)\ndef k():\n    pass\n")
    new, n = WARP_CALLS.rewrite(src, "case.py")
    assert n == 2 and WARP_CALLS.rewrite(new, "case.py") == (new, 0) and len(new.splitlines()) == len(src.splitlines())
    assert new.splitlines()[0] == "import warp as wp" + WARP_CALLS.IMPORT_TAIL + "  # noqa: F401"
    assert "_alloy_warp.module_name('é_mod')" in new and "@_alloy_warp.kernel(enable_backward=False, **OPTS)" in new


def test_upstream_is_the_exact_inverse_of_the_rewrite() -> None:
    """Every vendored file: rewriting its upstream form gives the file back (source pins hash the upstream form)."""
    from scripts_vendor import WARP_CALLS  # noqa: PLC0415

    for path in WARP_CALLS.vendored_files(VENDOR):
        text = path.read_text()
        assert "_alloy_warp" not in WARP_CALLS.upstream(text), path
        assert WARP_CALLS.rewrite(WARP_CALLS.upstream(text))[0] == text, path


def _stock_copy(dest: Path) -> Path:
    """The vendored trees rewritten back to stock form (`import newton`, `wp.kernel(module=...)`, bare module names):
    what a user's own newton / mujoco_warp at the same pins looks like."""
    from scripts_vendor import WARP_CALLS  # noqa: PLC0415

    for pkg in ("newton", "mujoco_warp"):
        for src in (VENDOR / pkg).rglob("*.py"):
            text = WARP_CALLS.upstream(src.read_text())
            for a, b in (("from alloy._vendor import newton", "import newton"),
                         ("from alloy._vendor import mujoco_warp", "import mujoco_warp"),
                         ("alloy._vendor.newton", "newton"), ("alloy._vendor.mujoco_warp", "mujoco_warp")):
                text = text.replace(a, b)
            assert "_alloy_warp" not in text, src
            out = dest / pkg / src.relative_to(VENDOR / pkg)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(text)
    return dest


PROBE = r'''
import importlib, pkgutil, sys, warp
import numpy as np
FIRST, OWN, PREFIX = @FIRST@, ("alloy._vendor.", "alloy_vendor__"), "alloy_vendor__"
SKIP = (".tests", ".examples", ".viewer", ".usd", "._src.utils.selection")     # optional parts, skipped in both
launched = []
for _name in ("launch", "launch_tiled"):                  # who launches which kernel (one copy's code, other's kernel?)
    def _spy(*a, _launch=getattr(warp, _name), **kw):
        launched.append((sys._getframe(1).f_globals.get("__name__", ""), kw.get("kernel", a[0] if a else None)))
        return _launch(*a, **kw)
    setattr(warp, _name, _spy)

def copy_of(m):
    return "alloy" if m.startswith("alloy._vendor.") else "user" if m.split(".")[0] in ("newton", "mujoco_warp") else ""

def walk(prefix):
    for top in ("newton", "mujoco_warp"):
        pkg = importlib.import_module(prefix + top)
        for info in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + "."):
            if not any(x in info.name for x in SKIP):
                try:
                    importlib.import_module(info.name)
                except Exception:
                    pass

MESH = "0 0 -0.02  0.3 0 -0.02  0 0.04 -0.02  0 0 0.02  0.3 0.04 0.02  0.3 0 0.02  0 0.04 0.02  0.3 0.04 -0.02"
ARM = f"""<mujoco><compiler angle="radian"/><asset><mesh name="hull" vertex="{MESH}"/></asset>
  <worldbody><body name="link1" pos="0 0 0.1"><joint name="shoulder" type="hinge" axis="0 1 0" range="-2 2"
  damping="2"/><geom type="mesh" mesh="hull" mass="1"/></body></worldbody>
  <actuator><position joint="shoulder" kp="200" kv="10"/></actuator></mujoco>"""

def alloy_step():                                          # a robot mesh beside a box under a cloth, through Alloy
    import alloy
    s = alloy.Scene(dt=0.01); s.add_ground(); s.add_robot("arm", mjcf=ARM, pos=(0, 0, 0.2))
    s.add_box("box", size=(0.1, 0.1, 0.1), pos=(0.3, 0.2, 0.05), static=True)
    s.add_cloth("sheet", size=(0.3, 0.3), pos=(0.15, 0, 0.45), resolution=5, mass=0.01)
    sim = alloy.build(s, device="cpu"); sim.step(2)
    return bool(sim.check()["ok"]), sorted({r["route"] for r in sim.routes().values()})

def user_step():                                           # the stock copy: collide (narrow phase) + XPBD cloth
    import newton
    import warp as wp
    b = newton.ModelBuilder(); b.add_ground_plane()
    box = b.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.1), wp.quat_identity()))
    b.add_shape_box(box, hx=0.05, hy=0.05, hz=0.05)
    v = np.array([[x, y, z] for x in (-.05, .05) for y in (-.05, .05) for z in (-.05, .05)], np.float32)
    f = [0, 1, 3, 0, 3, 2, 4, 6, 7, 4, 7, 5, 0, 4, 5, 0, 5, 1, 2, 3, 7, 2, 7, 6, 0, 2, 6, 0, 6, 4, 1, 5, 7, 1, 7, 3]
    hull = b.add_body(xform=wp.transform(wp.vec3(0.2, 0.0, 0.1), wp.quat_identity()))
    b.add_shape_mesh(hull, mesh=newton.Mesh(v, np.array(f, np.int32)))
    b.add_cloth_grid(pos=wp.vec3(-0.1, -0.1, 0.3), rot=wp.quat_identity(), vel=wp.vec3(0.0, 0.0, 0.0), dim_x=4,
                     dim_y=4, cell_x=0.05, cell_y=0.05, mass=0.01)
    model = b.finalize(device="cpu")
    s0, s1, control = model.state(), model.state(), model.control()
    solver = newton.solvers.SolverXPBD(model)
    for _ in range(2):
        solver.step(s0, s1, control, model.collide(s0), 0.01); s0, s1 = s1, s0
    return True, [newton.__file__]

order = ("alloy", "user") if FIRST == "alloy" else ("user", "alloy")
for side in order:
    walk("alloy._vendor." if side == "alloy" else "")
steps = [alloy_step() if side == "alloy" else user_step() for side in order]
reg = warp._src.context.user_modules
mixed, unshared, keys = [], [], set()
for name, m in list(reg.items()):
    ks = list(m.kernels.values()) + list(getattr(m, "_live_kernels", ()))
    keys |= {k.key for k in ks}
    copies = {copy_of(getattr(k.func, "__module__", "") or "") for k in ks} - {""}
    mixed += [name] if len(copies) > 1 else []
    unshared += [name] if "alloy" in copies and not name.startswith(OWN) else []
borrowed = sorted({f"{n}.{a}" for n, mod in list(sys.modules.items()) if copy_of(n) and mod is not None
                   for a, o in list(vars(mod).items())
                   if isinstance(o, warp.Kernel) and copy_of(o.func.__module__) not in (copy_of(n), "")})
runs = [(copy_of(c), k) for c, k in launched if isinstance(k, warp.Kernel) and copy_of(c)]
cross = sorted({k.key for side, k in runs if copy_of(k.func.__module__) not in (side, "")})
named = [k for side, k in runs if side == "user" and not k.module.name.startswith(k.func.__module__)]
twins = sorted({k.module.name for k in named if PREFIX + k.module.name in reg or PREFIX + k.key in keys})
print(repr(dict(steps=steps, mixed=mixed[:8], unshared=unshared[:8], borrowed=borrowed[:8], cross=cross[:8],
                launches={s: sum(1 for x, _ in runs if x == s) for s in ("alloy", "user")}, twins=twins[:8])))
'''


@pytest.mark.compile_heavy
@pytest.mark.timeout(600)   # the stock copy compiles cold in its subprocess (60 s cut it on a fresh VM)
@pytest.mark.parametrize("first", ["user", "alloy"])
def test_coexists_with_stock_engines_in_either_order(tmp_path: Path, first: str) -> None:
    """Alloy beside a stock-form newton / mujoco_warp (every submodule of both imported, ``first`` first), each copy
    stepping its own scene (the stock copy's collide + XPBD cloth; Alloy's robot mesh + cloth): no Warp module holds
    kernels of both copies, every module holding a vendored kernel has a name of Alloy's own (OSS-2's rule), no
    Python module of one copy holds the other's kernel, and neither copy launched the other's (WP8's review). The
    stock side must launch a named-module kernel whose twin Alloy also has, so the scene exercises the overlap.
    With unique keys unprefixed, only the naming rule fails here (decorator-form kernels already hash apart; the
    call-form clash itself is the synthetic test's control)."""
    stock = _stock_copy(tmp_path / "stock")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join((str(stock), str(ROOT))))
    code = PROBE.replace("@FIRST@", repr(first))
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True, timeout=1500)
    assert out.returncode == 0, out.stderr[-3000:]
    row = eval(out.stdout.strip().splitlines()[-1])           # noqa: S307 - our own print
    assert all(ok for ok, _ in row["steps"]) and str(stock) in row["steps"][0 if first == "user" else 1][1][0], row
    assert row["mixed"] == row["unshared"] == row["borrowed"] == row["cross"] == [], row
    assert min(row["launches"].values()) > 0 and row["twins"], row


SYNTH = '''import warp as wp


@wp.struct
class Pair:
    a: float
    b: float


@wp.func(module="coexist_synth_named")
def part(x: float):
    return x


@wp.func(module="coexist_synth_named")
def part(x: int):  # noqa: F811 - a Warp overload (found in the caller's locals: WP8's review)
    return float(x) * 2.0


def total(p: wp.array(dtype=Pair), out: wp.array(dtype=float)):
    i = wp.tid()
    out[i] = p[i].a + p[i].b + part(1) + part(1.0)


def make(kind):
    if kind == "unique":
        return wp.kernel(enable_backward=False, module="unique")(total)
    return wp.kernel(enable_backward=False, module="coexist_synth_named")(total)
'''

SYNTH_RUN = """
import importlib, warp as wp
def launch(pkg, kind):
    m = importlib.import_module(pkg + ".synth")
    k = m.make(kind)
    p = m.Pair(); p.a, p.b = 1.0, 2.0
    out = wp.zeros(1, dtype=float, device="cpu")
    wp.launch(k, dim=1, inputs=[wp.array([p], dtype=m.Pair, device="cpu"), out], device="cpu")
    return float(out.numpy()[0])
res = []
for first, second in {pairs!r}:
    for kind in ("unique", "named"):
        try:
            res.append((first, second, kind, launch(first, kind), launch(second, kind)))
        except Exception as exc:
            res.append((first, second, kind, type(exc).__name__))
print(repr(res))
"""


def _synth_packages(root: Path) -> None:
    from scripts_vendor import WARP_CALLS  # noqa: PLC0415

    for pkg, text in (("stock_a", SYNTH), ("stock_b", SYNTH), ("vend", WARP_CALLS.rewrite(SYNTH)[0])):
        (root / pkg).mkdir(parents=True)
        (root / pkg / "__init__.py").write_text("")
        (root / pkg / "synth.py").write_text(text)


def _synth_run(root: Path, pairs: list[tuple[str, str]]) -> list:
    env = dict(os.environ, PYTHONPATH=os.pathsep.join((str(root), str(ROOT))))
    code = SYNTH_RUN.format(pairs=pairs)
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr[-3000:]
    return eval(out.stdout.strip().splitlines()[-1])      # noqa: S307 - our own print


def test_the_rewrite_separates_call_form_and_named_kernels(tmp_path: Path) -> None:
    """Call-form `wp.kernel(module="unique")(template)` and named-module kernels (the 37 + 41 vendored sites whose
    hashed source does not include the decorator): two stock copies share them, so the second copy's launch fails on
    its own struct (the control, WP10's error); a copy through the rewrite launches with its own, in either order."""
    _synth_packages(tmp_path)
    control = _synth_run(tmp_path, [("stock_a", "stock_b")])
    assert [r[3] for r in control if r[2] == "unique"] == ["RuntimeError"], control
    rewritten = _synth_run(tmp_path, [("stock_a", "vend"), ("vend", "stock_a")])
    assert all(r[3:] == (6.0, 6.0) for r in rewritten), rewritten


def test_the_vendored_mujoco_version_check_reads_no_distribution(monkeypatch) -> None:
    """Patch 0010: the vendored SolverMuJoCo never reads the ``newton`` distribution's metadata (a user's own copy)."""
    from alloy._vendor.newton._src.solvers.mujoco import solver_mujoco as sm

    def boom(name: str):
        raise AssertionError(f"read distribution {name!r}")

    monkeypatch.setattr(sm.importlib_metadata, "distribution", boom)
    sm._warn_if_mujoco_versions_mismatch(None, None)
