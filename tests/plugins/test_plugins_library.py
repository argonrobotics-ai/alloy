# SPDX-License-Identifier: Apache-2.0
"""The built-in plug-ins pass the harness: quick checks (crash + laws) always, full checks marked slow."""

from __future__ import annotations

import pytest

import alloy.plugins as ap

pytestmark = pytest.mark.law


def _assert_pass(report):
    assert report.passed, report.summary()


@pytest.mark.parametrize("name", ["magnet", "ink", "hydraulic"])
def test_builtin_quick_check(name):
    report = ap.check(name, quick=True)
    _assert_pass(report)
    laws = [c.name for c in report.checks if c.group == "law"]
    assert len(laws) >= 2, laws


@pytest.mark.slow
@pytest.mark.parametrize("name", ["magnet", "ink", "hydraulic", "hydraulic_cylinder"])
def test_builtin_full_check(name):
    report = ap.check(name)
    _assert_pass(report)
    groups = {c.group for c in report.checks if not c.skipped}
    assert {"crash", "determinism", "independence"} <= groups


def test_harness_catches_a_broken_law():
    """A magnet whose force is twice too strong fails its inverse-fourth law (the harness is not a rubber stamp)."""
    from alloy.plugins.library import magnet

    class Strong(magnet.Magnet):
        """A magnet with double the moment it reports."""

        name = "strong_magnet_test"

        def setup(self, ctx):
            super().setup(ctx)
            self.moments = ctx.array([2.0 * m for m in self.config.moments])

    report = ap.check(Strong(), quick=True)
    failed = {c.name for c in report.failures}
    assert "inverse_fourth" in failed and "energy" in failed


BAD_HOOKS = '''
import math
import warp as wp
from alloy.plugins import ForcePlugin, TestScene, body_xform
from alloy.plugins.funcs import add_wrench


@wp.kernel
def _k(body: wp.array[wp.int32], f: wp.float32, body_f: wp.array[wp.spatial_vector]):
    w = wp.tid()
    add_wrench(body_f, body[w], wp.spatial_vector(f, 0.0, 0.0, 0.0, 0.0, 0.0))


class {name}(ForcePlugin):
    """Bad on purpose."""
    name = "{key}"
    summary = "bad"

    def setup(self, ctx):
        self.body = ctx.bodies("box")
        self.t = 0.0

    def apply(self, state, forces, dt):
        {body}
        wp.launch(_k, dim=self.ctx.world_count, inputs=[self.body, float(math.sin(20.0 * self.t))],
                  outputs=[forces.body_f], device=self.ctx.device)

    def test_scene(self):
        def build(b):
            b.add_shape_box(b.add_body(xform=body_xform(0.0, 0.0, 1.0), label="box"), hx=0.05, hy=0.05, hz=0.05)
        return TestScene(build=build, ground=False, gravity=(0.0, 0.0, 0.0), periods=5)
'''


def test_python_state_per_step_fails_graph_replay(tmp_path):
    src = tmp_path / "pulse.py"
    src.write_text(BAD_HOOKS.format(name="Pulse", key="pulse_test", body="self.t += dt"))
    report = ap.check(str(src), speed=False)
    failed = {c.name for c in report.failures}
    assert {"graph_replay", "hook_graph", "hook_determinism"} <= failed, report.summary()
    assert "crash" not in failed and "identical_worlds" not in failed


def test_host_readback_in_a_hook_is_reported(tmp_path):
    src = tmp_path / "peek.py"
    src.write_text(BAD_HOOKS.format(name="Peek", key="peek_test", body="self.t = float(state.body_q.numpy()[0, 2])"))
    report = ap.check(str(src), quick=True)
    crash = next(c for c in report.checks if c.name == "crash")
    assert not crash.passed and "wp.array.numpy()" in crash.detail, report.summary()
