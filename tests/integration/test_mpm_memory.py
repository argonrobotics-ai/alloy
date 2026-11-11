# SPDX-License-Identifier: Apache-2.0
"""The GPU memory of an MPM scene is judged before its first step for the mode that will run (SCR-WP12-3,
alloy.integration.mpm_memory): graphs when they fit, eager when only that fits, refused when neither does. CPU tests on
stand-in built scenes and devices; the laws on an L4 are in tests/integration/test_mpm_memory_gpu.py."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest

from alloy.core.errors import CapacityError
from alloy.integration import mpm_memory as M

GB = 2**30
DOUGH, DIG = 17_256, 15_360                   # MPM particles per world: a dough_ball, soil_dig's bed


@dataclass
class _Built:
    entries: list
    world_count: int
    model: Any
    report: Any
    precision_pairs: list = field(default_factory=list)


def _built(worlds: int, per_world: int, *, bodies: int = 1, device="cuda:0", capability="mpm") -> _Built:
    n = worlds * per_world
    entries = [SimpleNamespace(capability=capability, particles=tuple(range(n)), solver_kwargs={}),
               SimpleNamespace(capability="mujoco", particles=(), solver_kwargs={"jacobian": "sparse"})]
    decisions = {"soil": SimpleNamespace(entry=capability), "tray": SimpleNamespace(entry=None)}
    b = _Built(entries, worlds, SimpleNamespace(device=device, body_count=bodies),
               SimpleNamespace(decisions=decisions, scene_effects=[]))
    b._plan = "assembly plan"                   # set outside the dataclass fields, as the assembler does
    return b


@pytest.fixture()
def l4(monkeypatch):
    """An L4 with the model on it: 21.4 GB free before a 256-world dough model (~2.7 GB), 18.7 after."""
    dev = SimpleNamespace(is_cuda=True, free_memory=int(18.7 * GB), name="NVIDIA L4", __str__=lambda s: "cuda:0")
    monkeypatch.setattr("warp.get_device", lambda *a: dev)
    monkeypatch.setattr(M, "free_bytes", lambda d: d.free_memory)
    dev.trims = []                                 # the graph pool, trimmed after each collection
    monkeypatch.setattr("alloy.parallel.counters.trim_graph_pool", dev.trims.append)
    return dev


def test_a_scene_that_fits_as_graphs_comes_back_unchanged(l4) -> None:
    b = _built(8, DOUGH)
    assert M.plan(b, True) is b and M.eager_reason(b) is None     # the graph path is not touched


def test_256_dough_worlds_run_eager_instead_of_being_refused(l4) -> None:
    b = _built(256, DOUGH)
    out = M.plan(b, True)
    why = M.eager_reason(out)
    assert out is not b and M.eager_reason(b) is None             # a marked copy; the caller's scene is unchanged
    assert out._plan == "assembly plan" and out.entries is b.entries
    assert "soil: 4,417,536 particles over 256 world(s)" in why and "runs without graphs" in why
    assert f"about {M.predict(256 * DOUGH, graph=False) / GB:.1f} GB" in why


def test_soil_dig_at_64_worlds_runs_eager(l4) -> None:
    """Refused before SCR-WP12-3 (its graphs need ~30 GB); its eager step measured 2.3 GB at 64 worlds."""
    assert M.eager_reason(M.plan(_built(64, DIG), True))


def test_a_scene_that_fits_in_neither_mode_is_refused_with_the_eager_numbers(l4) -> None:
    with pytest.raises(M.MPMMemoryError) as e:
        M.plan(_built(1024, DIG), True)
    err = e.value
    assert isinstance(err, CapacityError) and err.need == M.predict(1024 * DIG, graph=False)
    assert err.free == l4.free_memory and 256 <= err.fits < 1024
    text = str(err)
    assert "soil: 15,728,640 particles over 1024 world(s)" in text and "even without CUDA graphs" in text
    assert "as CUDA graphs" in text and "GB is free" in text and "world(s) of this scene fit" in text
    assert "fewer worlds" in err.fix


def test_without_graphs_asked_for_only_the_eager_step_is_judged(l4) -> None:
    b = _built(256, DOUGH)
    assert M.plan(b, False) is b                                  # the debug profile: eager fits, nothing to switch
    with pytest.raises(M.MPMMemoryError) as e:
        M.plan(_built(1024, DIG), False)
    assert "as CUDA graphs" not in str(e.value)


@pytest.mark.parametrize("after", [18.7, 11.8])
def test_a_scene_is_judged_after_a_garbage_collection(l4, monkeypatch, after) -> None:
    """A dropped simulation is a reference cycle that keeps its device memory until a full collection (WP11's GPU
    line: 11.8 GB free before 256 dough worlds, after 35 GPU tests in one process). The reading after the collection
    decides, also for a scene that fits in the first one (SCR-OSS22-9: graphs started in a sweep's leftovers)."""
    reads, collects = iter([int(11.8 * GB), int(after * GB)]), []
    monkeypatch.setattr(M, "free_bytes", lambda d: next(reads))
    monkeypatch.setattr(M.gc, "collect", lambda: collects.append(1))
    if after < 12:
        with pytest.raises(M.MPMMemoryError, match="11.8 GB is free"):
            M.plan(_built(256, DOUGH), True)
    else:
        assert "18.7 GB is free" in M.eager_reason(M.plan(_built(256, DOUGH), True))
    assert collects == [1]
    reads = iter([int(11.8 * GB), int(18.7 * GB)])
    b = _built(24, DOUGH)                                           # 12.9 GB as graphs: over the first reading
    assert M.plan(b, True) is b and collects == [1, 1]              # collected before it is judged: graphs stay
    assert len(l4.trims) == 2                                       # and the collected graphs' pool memory handed back
    monkeypatch.setenv(M.ENV, "off")
    assert M.plan(_built(24, DOUGH), True) is not None and collects == [1, 1]


def test_room_to_step_refuses_an_mpm_scene_whose_eager_step_has_no_room(l4) -> None:
    """After a step graph ran out of memory the period steps eagerly at once: with no room for that, the clean
    refusal, before any Warp native runs on a full device (SCR-OSS22-9)."""
    b = _built(64, 7_688, bodies=0)                                 # the sand pile: 492,032 particles, 0.47 GB
    assert M.room_to_step(b, "cuda:0", free=GB) is None
    assert M.room_to_step(b, "cuda:0") is None                      # reads the device: 18.7 GB
    with pytest.raises(M.MPMMemoryError) as e:
        M.room_to_step(b, "cuda:0", free=GB // 4)
    text = str(e.value)
    assert "its CUDA step graph ran out of memory on " in text and "to step on without graphs" in text
    assert "soil: 492,032 particles over 64 world(s)" in text and "0.5 GB more" in text and "0.2 GB is free" in text
    assert "fewer worlds" in e.value.fix and 0 < e.value.fits < 64 and e.value.free == GB // 4
    assert e.value.need == 492_032 * M.BYTES_STEP


@pytest.mark.parametrize("case", ["cpu", "no mpm", "off"])
def test_room_to_step_asks_nothing_on_the_cpu_without_mpm_or_when_switched_off(l4, monkeypatch, case) -> None:
    if case == "cpu":
        l4.is_cuda = False
    if case == "off":
        monkeypatch.setenv(M.ENV, "off")
    b = _built(1024, DIG, capability="sph" if case == "no mpm" else "mpm")
    assert M.room_to_step(b, "cuda:0", free=0) is None


@pytest.mark.parametrize("case", ["cpu", "no mpm", "off"])
def test_no_check_on_the_cpu_without_mpm_or_when_switched_off(l4, monkeypatch, case) -> None:
    if case == "cpu":
        l4.is_cuda = False
    if case == "off":
        monkeypatch.setenv(M.ENV, "off")
    b = _built(1024, DIG, capability="sph" if case == "no mpm" else "mpm")
    assert M.plan(b, True) is b


def test_predict_eager_below_graphs_and_growing_with_the_particles() -> None:
    assert M.predict(10**6, graph=False) < M.predict(10**6, moving=False) < M.predict(10**6, moving=True)
    assert M.predict(2_000, graph=False) > M.predict(1_000, graph=False) > M.FIXED_BYTES


def test_the_scene_graph_decision_follows_the_plan(monkeypatch) -> None:
    from alloy.integration import wiring
    from alloy.integration.profiles import profile

    monkeypatch.setattr(wiring.wp, "get_device", lambda *a: SimpleNamespace(is_cuda=True))
    b = _built(8, DOUGH)
    assert wiring._scene_graphs(b, profile("standard")) == (True, False)
    setattr(b, M.EAGER_ATTR, "too big as graphs")
    assert wiring._scene_graphs(b, profile("standard")) == (False, False)
    assert wiring._scene_graphs(b, profile("debug")) == (False, False)


@pytest.mark.compile_heavy
@pytest.mark.timeout(600)
def test_wire_carries_the_eager_marker_to_the_built_it_hands_on(monkeypatch) -> None:
    """WP4's review: every copy ``wire`` makes after ``plan`` must keep the marker, or ``_scene_graphs`` turns graphs
    back on (~10x the memory) while the notes still say "graphs off"."""
    from alloy.core.registry import isolated_registry
    from alloy.integration import wiring
    from alloy.integration.built_copy import replace_built
    from alloy.integration.profiles import profile
    from alloy.testing import integration as I
    from alloy.testing import scene as S

    def marked(built, graph):
        out = replace_built(built)
        setattr(out, M.EAGER_ATTR, "too big as graphs (test)")
        return out

    monkeypatch.setattr(M, "plan", marked)
    card, lib = {n: (c, lb) for n, c, lb in S.mixed_scenes()}["box_on_table"]
    with isolated_registry(empty=True):
        built = I.assemble(card, lib, "cpu", 2)
        parts = wiring.wire(built, "standard")
    assert M.eager_reason(parts.built) == "too big as graphs (test)" and M.eager_reason(built) is None
    assert "graphs off: too big as graphs (test)" in parts.notes
    monkeypatch.setattr(wiring.wp, "get_device", lambda *a: SimpleNamespace(is_cuda=True))
    assert wiring._scene_graphs(parts.built, profile("standard")) == (False, False)


@pytest.mark.parametrize("helper", ["_prepare_entries", "_route_entries", "_graph_entries"])
def test_every_entry_copy_in_wire_keeps_the_eager_marker(helper) -> None:
    """The copies ``wire`` makes after ``plan`` (precision pairs, the ordered MuJoCo route, the graph grid) go through
    ``replace_built``; a plain ``dataclasses.replace`` would drop the marker (WP4's review)."""
    from alloy.integration import wiring
    from alloy.integration.profiles import profile

    b = _built(2, DOUGH, capability="sph")                        # entries the helpers copy without changing
    b.precision_pairs, b.entries = ["pair"], b.entries[:1]        # no MuJoCo entry to adjust for the pair
    setattr(b, M.EAGER_ATTR, "too big as graphs (test)")
    call = {"_prepare_entries": lambda: wiring._prepare_entries(b, "mujoco", profile("standard"), []),
            "_route_entries": lambda: wiring._route_entries(b, profile("standard", mjw_route=True), []),
            "_graph_entries": lambda: wiring._graph_entries(b, [])}[helper]
    out = call()
    assert out is not b and M.eager_reason(out) == "too big as graphs (test)"

