# SPDX-License-Identifier: Apache-2.0
"""The CPU route without its C step (F18: a box without a C compiler): the build or load failure falls back to the
numpy step with one user-facing warning that names the fix, whatever the failure, and the bodies still step.
``ALLOY_NO_EXT=1`` switches the C step off the same way (SCR-OSS1-8: release CI runs the cpu_route tests both
ways)."""

from __future__ import annotations

import numpy as np
import pytest

from alloy._vendor import newton

import alloy.route._mjc as mjc
import alloy.route._mjc.build as mjc_build
import alloy.route.mjc_core as mc
import alloy.route.mjc_core_cabi as cabi
from alloy.bully.direct import DirectSim
from alloy.route.mjc_core import MjcCoreSolver
from alloy.route.mjc_core_config import CPU_ROUTE_KEY
from alloy.scene.builtin_plugins import builtin_capabilities

pytestmark = pytest.mark.cpu_route

XML = """<mujoco><option timestep="0.002"/><worldbody><geom type="plane" size="1 1 .1"/>
 <body pos="0 0 .2"><freejoint/><geom type="box" size=".05 .05 .05" mass="1"/></body>
</worldbody></mujoco>"""


def _route() -> MjcCoreSolver:
    t = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(t)
    t.add_mjcf(XML, parse_visuals=False)
    b = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(b)
    b.replicate(t, copy_count=2)
    model = b.finalize(device="cpu")
    solver = next(c for c in builtin_capabilities() if c.name == "mujoco").factory(model, {CPU_ROUTE_KEY: "mjc"})
    assert isinstance(solver, MjcCoreSolver)
    return solver


def _forget(monkeypatch) -> None:
    """No extension loaded or tried yet, no warning given yet, ALLOY_NO_EXT unset."""
    monkeypatch.setattr(mjc, "_LIB", None)
    monkeypatch.setattr(mjc, "_TRIED", False)
    monkeypatch.setattr(mjc, "reason", "")
    monkeypatch.setattr(mc, "_WARNED", set())
    monkeypatch.setattr(cabi, "_FAILURE", "")
    monkeypatch.delenv(mjc.NO_EXT_ENV, raising=False)


@pytest.fixture
def fresh(monkeypatch, tmp_path):
    """No loaded extension, no prebuilt or cached library (an empty cache), no warning given yet."""
    _forget(monkeypatch)
    monkeypatch.setattr(mjc_build, "HERE", tmp_path / "src")             # no prebuilt library beside the sources
    monkeypatch.setattr(mjc_build, "cache_root", lambda: tmp_path / "cache")
    monkeypatch.setattr(mjc_build, "build_key", lambda *a: "k")


def _no_compiler():
    raise mjc_build.BuildError("no C compiler (set CC or install cc/gcc/clang)")


def _warnings(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.name == "alloy.api" and "fast C step" in r.getMessage()]


def test_without_a_compiler_the_route_steps_on_numpy_and_says_how_to_fix_it(fresh, monkeypatch, caplog):
    """A missing compiler: the route takes the numpy step and gives one user-facing warning with the fix (alloy.api,
    printed to stderr when no handler shows it), not again for a second route; the box falls and lands, finite."""
    monkeypatch.setattr(mjc_build, "compiler", _no_compiler)
    solver = _route()
    assert solver.backend == "numpy" and "no C compiler" in solver.backend_reason
    msgs = _warnings(caplog)
    assert len(msgs) == 1 and "install a C compiler" in msgs[0] and "numpy step" in msgs[0], msgs
    assert _route().backend == "numpy" and len(_warnings(caplog)) == 1
    sim = DirectSim(solver.model, solver, control_dt=0.016, substeps=8, collide=False, export_contacts=True)
    for _ in range(25):
        sim.step_period()
    z = sim.state.body_q.numpy()[:, 2]
    assert np.isfinite(z).all() and np.allclose(z, 0.05, atol=5e-3), z


@pytest.mark.parametrize("error", [PermissionError(13, "Permission denied", "/nonexistent/cache/alloy"),
                                   ValueError("a broken mujoco wheel")])
def test_any_build_failure_falls_back(fresh, monkeypatch, caplog, error):
    """Not a BuildError (an unwritable cache, a broken install): the route still takes the numpy step, names the
    failure and keeps naming it."""
    def fail():
        raise error
    monkeypatch.setattr(mjc_build, "compiler", lambda: ("/usr/bin/cc", "cc 12"))
    monkeypatch.setattr(mjc_build, "mujoco_paths", lambda: ("3.3.0", mjc_build.HERE, mjc_build.HERE / "libmujoco.so"))
    monkeypatch.setattr(mjc_build, "cache_root", fail)
    solver = _route()
    assert solver.backend == "numpy" and type(error).__name__ in solver.backend_reason
    assert _route().backend_reason == solver.backend_reason                         # kept, not "load() returned None"
    msgs = _warnings(caplog)
    assert len(msgs) == 1 and "reinstall alloy-engine" in msgs[0], msgs


def test_with_no_log_handler_the_warning_reaches_stderr(fresh, monkeypatch, capsys):
    """No handler but Alloy's NullHandler (a plain script): the warning is printed, so the fallback is not silent."""
    import logging

    root = logging.getLogger()
    monkeypatch.setattr(root, "handlers", [])
    monkeypatch.setattr(mjc_build, "compiler", _no_compiler)
    assert _route().backend == "numpy"
    err = capsys.readouterr().err
    assert err.count("alloy: CPU route: the fast C step") == 1 and "install a C compiler" in err, err


def test_the_fix_names_the_compiler_when_one_is_missing_or_fails():
    assert "apt install gcc" in mc.fallback_fix("no C compiler (set CC or install cc/gcc/clang)", "linux")
    assert "apt install gcc" in mc.fallback_fix("compiling liballoy_mjc failed (/usr/bin/cc): error", "linux")
    assert "xcode-select --install" in mc.fallback_fix("no C compiler: /usr/bin/cc --version failed (xcrun)", "darwin")
    assert "reinstall" in mc.fallback_fix("built for MuJoCo 3.2.0, the installed one is 3.3.0")
    assert mc.fallback_fix("ALLOY_NO_EXT=1 switches the compiled CPU-route extension off") == \
        "unset ALLOY_NO_EXT for the fast C step"


@pytest.mark.parametrize("off", [True, False], ids=["ALLOY_NO_EXT=1", "extension"])
def test_the_fallback_warning_comes_exactly_when_the_extension_is_off(monkeypatch, caplog, off):
    """ALLOY_NO_EXT=1: the route steps on numpy with the missing-extension warning, once, naming the variable, and
    nothing is built. Unset (this machine has a compiler, as the gate and release CI do): the C step, no warning."""
    _forget(monkeypatch)
    if off:
        monkeypatch.setenv(mjc.NO_EXT_ENV, "1")
        monkeypatch.setattr(mjc, "library_path", _no_compiler)                # would raise: never reached
    solver = _route()
    assert solver.backend == ("numpy" if off else "c"), solver.backend_reason
    msgs = _warnings(caplog)
    if not off:
        assert msgs == [] and solver.backend_reason == ""
        return
    assert "ALLOY_NO_EXT=1" in solver.backend_reason and mjc.load() is None
    assert len(msgs) == 1 and "numpy step" in msgs[0] and "unset ALLOY_NO_EXT" in msgs[0] and ".." not in msgs[0], msgs
    assert _route().backend == "numpy" and len(_warnings(caplog)) == 1
