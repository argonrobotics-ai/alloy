# SPDX-License-Identifier: Apache-2.0
"""OSS-7 SCR-SOFT-3-2: the check's world count fits the device memory. Fit arithmetic and wording on a stand-in
scene; on CPU (free memory faked) a check asked for 2 worlds runs 1 and says why; a simulation that cannot be
allocated is a health error that says to pass fewer worlds."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import alloy.check
from alloy.cards import types as ct
from alloy.check import memory

TABLE = ct.Placement("table_top_plywood", (0.0, 0.0, -0.009, 0.0, 0.0, 0.0, 1.0))
CUBE = ct.Placement("cube_beech_30", (0.0, 0.0, 0.015, 0.0, 0.0, 0.0, 1.0))


def _built(worlds: int, mpm_per_world: int):
    model = SimpleNamespace(shape_count=2 * worlds, joint_dof_count=0, body_count=0)
    entries = [SimpleNamespace(capability="mujoco", particles=()),
               SimpleNamespace(capability="mpm", particles=range(mpm_per_world * worlds))]
    return SimpleNamespace(model=model, world_count=worlds, entries=entries)


def test_mpm_particles_set_the_world_count():
    built = _built(64, 20_000)
    parts, particles = memory.per_world(built)
    assert particles == {"mpm": 20_000} and parts["mpm"] == 20_000 * memory.BYTES_PER_PARTICLE["mpm"]
    need = sum(parts.values())
    free = 20e9
    fit = memory.fit_worlds(built, 64, "cuda:0", free=free)
    assert fit.worlds == int((memory.USABLE * free - memory.RESERVE) // need) and fit.capped
    assert fit.why.startswith(f"memory: {fit.worlds} of the 64 worlds asked fit (about ")
    assert "mpm 20,000 particles; 20 GB free on cuda:0" in fit.why
    assert memory.fit_worlds(built, 64, "cuda:0", free=1e6).worlds == 1           # at least one world
    rigid = memory.fit_worlds(_built(64, 0), 64, "cuda:0", free=free)
    assert rigid.worlds == 64 and not rigid.capped and rigid.why.startswith("asked: 64 worlds fit")
    assert memory.fit_worlds(built, 2, "cpu").why == "asked: 2 worlds (CPU memory is not sized)"


def test_check_runs_the_worlds_that_fit(monkeypatch):
    # one cube world needs ~4 MB here; free memory for one world but not two
    monkeypatch.setattr(memory, "RESERVE", 0)
    monkeypatch.setattr("alloy.check.api.free_bytes", lambda device: 6e6 / memory.USABLE)
    scene = ct.SceneCard(robots=(), objects=(TABLE, CUBE))
    report = alloy.check(scene, worlds=2, seconds=0.1, device="cpu")
    assert report.ok, report.summary()
    s = report.settings
    assert (s["worlds"], s["worlds_asked"]) == (1, 2) and s["worlds_why"].startswith("memory: 1 of the 2 worlds")
    assert report.metrics["scene"]["worlds"] == 1 and report.metrics["memory"]["worlds"] == 1
    (f,) = [f for f in report.findings if f.check == "health"]
    assert f.level == "info" and f.message.startswith("ran 1 of the 2 worlds asked, as many as fit in free GPU memory")
    assert "larger GPU" in f.fix
    assert report.checks["health"] == "pass"


def test_out_of_memory_build_is_a_finding(monkeypatch):
    def oom(*a, **k):
        raise RuntimeError("Failed to allocate 282722304 bytes on device 'cuda:0'")

    monkeypatch.setattr("alloy.integration.wiring.build_simulation", oom)
    report = alloy.check(ct.SceneCard(robots=(), objects=(TABLE, CUBE)), worlds=1, seconds=0.1, device="cpu")
    (f,) = report.errors()
    assert f.check == "health" and "Failed to allocate" in f.message and "fewer worlds" in f.fix
    assert report.checks["rest_drift"].startswith("skipped: the simulation did not run")
    assert report.settings["worlds_why"] == "asked: 1 world (CPU memory is not sized)"


@pytest.mark.parametrize("b, text", [(3e6, "3 MB"), (1.25e9, "1.2 GB"), (21.8e9, "22 GB"), (512, "512 B")])
def test_sizes_read_well(b, text):
    assert memory._size(b) == text
