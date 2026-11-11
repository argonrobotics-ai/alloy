# SPDX-License-Identifier: Apache-2.0
"""Engine errors get remedies; health reports become plain words with fixes."""

from __future__ import annotations

import json

import numpy as np
import pytest

from alloy.api import errors as E
from alloy.api import health as H
from alloy.core.errors import AssemblyError, CapacityError, CardError, CouplingError, PrecisionError, RouteError


@pytest.mark.parametrize("exc, needle", [
    (CardError("cube", "shape.size", "mesh x is inside out"), "winding"),
    (RouteError("no route for rod"), "solver"),
    (CapacityError("towel: 9000 SPH particles > budget"), "particles"),
    (PrecisionError("world 3 INVALID"), "dt"),
    (AssemblyError("world_count must be >= 1"), "worlds"),
    (CouplingError("scene has no dynamic entry to step"), "dynamic object"),
    (ModuleNotFoundError("No module named 'warp'"), "pip install"),
])
def test_remedies(exc, needle):
    assert needle in E.remedy_for(exc)


def test_wrapped_keeps_the_cause_and_the_field():
    with pytest.raises(E.BuildError) as info:
        with E.wrapped("build"):
            raise CardError("cube", "material", "unknown material 'x'")
    err = info.value
    assert isinstance(err.__cause__, CardError) and err.field == "cube.material"
    assert "build failed" in err.message and err.fix
    assert json.loads(json.dumps(err.to_dict()))["field"] == "cube.material"


def test_api_errors_pass_through_unchanged():
    original = E.SceneError("bad", "do this")
    with pytest.raises(E.SceneError) as info:
        with E.wrapped("build"):
            raise original
    assert info.value is original


def test_every_health_flag_has_a_remedy():
    from alloy.precision.types import HealthFlags

    assert {f.name for f in HealthFlags} <= set(H.REMEDIES)


def test_summary_of_an_invalid_world():
    from alloy.precision.types import HealthFlags, HealthReport

    rep = HealthReport.ok(3)
    rep.severity[1] = 2
    rep.flags[1] = int(HealthFlags.PEN_FAIL | HealthFlags.LATCHED)
    rep.first_bad_step[1] = 17
    rep = HealthReport(**{**vars(rep), "invalid_worlds": (1,)})
    out = H.summary(rep, 3)
    assert out["ok"] is False and out["invalid_worlds"] == [1]
    w = out["worlds"][1]
    assert w["severity"] == "INVALID" and w["flags"] == ["PEN_FAIL", "LATCHED"] and w["first_bad_step"] == 17
    assert any("overlap" in fix for fix in w["fixes"]) and "problem" in w
    json.dumps(out)
    assert H.summary(None, 2)["monitored"] is False
    assert np.isnan(rep.max_pen_m).all()


def test_retired_liquid_particles_are_reported():
    from types import SimpleNamespace

    class _SPH:
        def add_emitter(self, *a):  # marks a liquid solver
            pass

        def report(self):
            return SimpleNamespace(escaped=np.array([0, 3]))

    engine = SimpleNamespace(solvers=(object(), _SPH()))
    assert H.retired(engine) == [0, 3] and H.retired(SimpleNamespace(solvers=())) is None
    out = H.add_retired({"ok": True, "problems": []}, H.retired(engine))
    assert out["ok"] is True and out["retired_particles"] == [0, 3]
    assert [p["world"] for p in out["problems"]] == [1] and "retired" in out["problems"][0]["what"]
    json.dumps(out)


def test_an_engine_error_with_its_own_fix_keeps_it():
    """WP12's UnsupportedPairError carries the nearest supported alternative for its route pair as ``.fix``."""
    class PairError(RouteError):
        def __init__(self, msg: str, fix: str) -> None:
            super().__init__(msg)
            self.fix = fix

    own = E.wrap(PairError("cloth + mpm: no coupling", "make the box rigid=True"), "build")
    assert own.fix == "make the box rigid=True" and isinstance(own, E.BuildError)
    assert E.wrap(PairError("x", "  "), "build").fix == E.remedy_for(RouteError("x"))     # blank: the kind's remedy
    assert E.wrap(RouteError("no route for rod"), "build").fix == E.remedy_for(RouteError("no route for rod"))
    with pytest.raises(E.BuildError) as err, E.wrapped("build"):
        raise PairError("cloth + mpm", "use add_object(..., rigid=True)")
    assert err.value.fix == "use add_object(..., rigid=True)"


_INT32 = ValueError("Array shapes must not exceed the maximum representable value of a signed 32-bit integer, "
                    "got (16384, 9000000)")
_OOM = RuntimeError("Failed to allocate 237895680 bytes on device 'cuda:0'")


@pytest.mark.parametrize("exc, what", [(_INT32, "32-bit"), (_OOM, "out of memory")])
def test_world_limit_errors_become_build_errors(exc, what):
    err = E.limit_error(exc, 4096)
    assert isinstance(err, E.BuildError) and what in err.message and "4,096" in err.message
    assert "fewer worlds" in err.fix and "alloy.estimate" in err.fix


def test_other_value_and_runtime_errors_are_not_translated():
    assert E.limit_error(ValueError("bad shape"), 8) is None
    assert E.limit_error(RuntimeError("Warp CUDA error 2: out of memory (in function wp_alloc_device_async)"), 8)
    own = E.BuildError("the scene does not fit: out of memory on cuda:0", "run fewer MPM particles")
    assert E.limit_error(own, 8) is None                            # an Alloy error keeps its own message and fix
    with pytest.raises(E.BuildError) as info, E.world_limits(8):
        raise own
    assert info.value is own
    with pytest.raises(RuntimeError, match="boom"):
        with E.world_limits(8):
            raise RuntimeError("boom")


@pytest.mark.parametrize("exc", [_INT32, _OOM])
def test_alloy_build_translates_world_limit_failures(monkeypatch, exc):
    import alloy
    from alloy.api import _build

    def boom(*args, **kwargs):
        raise exc

    monkeypatch.setattr(_build, "build_block", boom)
    scene = alloy.Scene(dt=0.01)
    scene.add_ground()
    scene.add_box("b", size=(0.1, 0.1, 0.1), pos=(0, 0, 0.5), mass=1.0)
    with pytest.raises(alloy.BuildError) as info:
        alloy.build(scene, worlds=16384, device="cpu")
    assert info.value.__cause__ is exc and "16,384" in info.value.message
    assert "alloy.estimate" in info.value.fix
