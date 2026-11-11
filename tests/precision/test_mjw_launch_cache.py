# SPDX-License-Identifier: Apache-2.0
"""``mjw_launch._bind``'s signature cache (SCR-WP8-5 Ask 1; WP5 review criteria) on CPU.

- The signature is computed at install from the ``wp.launch`` being wrapped, dropped by ``uninstall()``, and
  recomputed by a reinstall over a ``wp.launch`` with a different signature (a stale one would hand the new launch
  keyword arguments it does not take).
- The bound ``Launch`` is the one an uncached ``inspect.signature(...).bind`` + ``apply_defaults`` gives: same
  fields, same ``kwargs`` keys (``launch_as_is`` passes them back to Warp).
"""

from __future__ import annotations

import inspect
import sys
import types

import pytest
import warp as wp

from alloy.precision import mjw_launch, mjw_route

FAKE = "alloy_fake_mjw_cache"
CORE = ("kernel", "dim", "inputs", "outputs", "block_dim")


@wp.kernel
def _write_one(out: wp.array[float]):
    out[wp.tid()] = 1.0


class _Model:  # weak-referenceable stand-in for a MuJoCo Warp Model
    pass


@pytest.fixture
def fake_mjw():
    """A one-stage fake MuJoCo Warp package; the module state starts and ends uninstalled (as test_mjw_route)."""
    mjw_route.uninstall()
    mod = types.ModuleType(FAKE)
    mod.factory = lambda flag: _write_one
    mod.stage = lambda m, out: wp.launch(mod.factory(False), dim=out.shape[0], outputs=[out], device="cpu")
    sys.modules[FAKE] = mod
    try:
        yield mod
    finally:
        mjw_launch.uninstall()
        del sys.modules[FAKE]


def _install(seen: list) -> None:
    def handler(args, launch):
        seen.append(launch)
        mjw_launch.launch_as_is(launch)

    mjw_launch.substitute(FAKE, "factory", handler)
    mjw_launch.scope(FAKE, "stage")


def _run(mod, seen: list) -> tuple[mjw_launch.Launch, list]:
    model = _Model()
    mjw_launch.enable(model)
    out = wp.zeros(3, dtype=float, device="cpu")
    mod.stage(model, out)
    return seen[-1], out.numpy().tolist()


def test_cache_refreshes_after_uninstall_and_reinstall(monkeypatch, fake_mjw):
    # monkeypatch before fake_mjw: the fixture's uninstall puts back launch_v2, then monkeypatch the real one
    real, seen, calls = wp.launch, [], []
    _install(seen)
    assert mjw_launch._STATE["sig"][0] is real  # taken at install, from the launch it wraps
    first, out = _run(fake_mjw, seen)
    assert out == [1.0, 1.0, 1.0]
    assert set(first.kwargs) == set(inspect.signature(real).parameters) - set(CORE)
    mjw_launch.uninstall()
    assert mjw_launch._STATE["sig"] is None and wp.launch is real

    def launch_v2(kernel, dim, inputs=(), outputs=(), block_dim=None, *, device=None, tag="v2"):
        calls.append(tag)
        return real(kernel, dim=dim, inputs=list(inputs), outputs=list(outputs), device=device)

    monkeypatch.setattr(wp, "launch", launch_v2)
    _install(seen)
    assert mjw_launch._STATE["sig"][0] is launch_v2
    second, out = _run(fake_mjw, seen)
    assert set(second.kwargs) == {"device", "tag"} and second.kwargs["tag"] == "v2"
    assert calls == ["v2"] and out == [1.0, 1.0, 1.0]  # launch_as_is went through the new launch with its keys
    mjw_launch.uninstall()
    assert wp.launch is launch_v2


def test_stale_entry_is_recomputed(fake_mjw):
    _install([])
    stale = inspect.signature(lambda kernel, dim, inputs, outputs, block_dim: None)
    mjw_launch._STATE["sig"] = (object(), stale)  # keyed on identity: another callable's entry is never used
    assert mjw_launch._launch_signature() == inspect.signature(mjw_launch._STATE["launch"])
    assert mjw_launch._STATE["sig"][0] is mjw_launch._STATE["launch"]


@pytest.mark.parametrize("args,kwargs", [
    ((3,), {"outputs": ["o"]}),
    ((3, ["i"], ["o"]), {}),
    ((), {"dim": (2, 4), "inputs": ["i"], "outputs": ["o"], "block_dim": 64, "device": "cpu"}),
    ((5,), {"inputs": ["i"], "record_cmd": True, "max_blocks": 2}),
])
def test_bound_launch_matches_uncached_bind(fake_mjw, args, kwargs):
    _install([])
    got = mjw_launch._bind(_write_one, args, kwargs)
    b = inspect.signature(mjw_launch._STATE["launch"]).bind(_write_one, *args, **kwargs)
    b.apply_defaults()
    a = b.arguments
    want = mjw_launch.Launch(a["dim"], list(a["inputs"] or []), list(a["outputs"] or []), a["block_dim"],
                             {k: v for k, v in a.items() if k not in CORE}, ("out",), _write_one)
    assert got == want and list(got.kwargs) == list(want.kwargs)
