# SPDX-License-Identifier: Apache-2.0
"""Both array backends offer the same basic math, so a ``checks.custom`` function runs on either.

The NumPy backend had no ``sqrt`` (nor ``abs``, ``exp``, ...): a custom check written ``s.xp.sqrt(...)`` died with
"AttributeError: 'NumpyBackend' object has no attribute 'sqrt'", and no page listed what ``xp`` offers.
"""
from __future__ import annotations

import numpy as np
import pytest

import alloy
from alloy.envs import Task, VectorEnv
from alloy.envs import checks as C
from alloy.envs.backend import MATH, NumpyBackend, TorchBackend

SLIDE = """<mujoco><worldbody><body name="cart" pos="-1 0 0.05"><joint name="x" type="slide" axis="1 0 0" range="-0.1 0.1"/>
  <geom type="box" size="0.05 0.05 0.05" mass="1"/></body></worldbody>
  <actuator><position joint="x" kp="100" kv="20"/></actuator></mujoco>"""
X = np.array([[0.25, 0.5, 0.75], [0.1, 0.2, 0.9]], np.float32)
WANT = {"minimum": np.minimum(X, 0.5), "maximum": np.maximum(X, 0.5), "sum": X.sum(-1), "mean": X.mean(-1)}


def _backends() -> list:
    out = [pytest.param(lambda: NumpyBackend("cpu"), id="numpy")]
    return out + [pytest.param(lambda: TorchBackend("cpu"), id="torch",
                               marks=pytest.mark.skipif(not _torch(), reason="PyTorch not installed"))]


def _torch() -> bool:
    from alloy.envs.backend import has_torch

    return has_torch()


@pytest.mark.parametrize("make", _backends())
@pytest.mark.parametrize("name", MATH)
def test_every_shared_operation_gives_numpys_values(make, name):
    xp = make()
    args = (xp.asarray(X), 0.5) if name in ("minimum", "maximum") else (xp.asarray(X),)
    got = xp.to_numpy(getattr(xp, name)(*args))
    want = WANT[name] if name in WANT else getattr(np, name)(X)
    np.testing.assert_allclose(got, want, rtol=1e-6)


@pytest.mark.timeout(300)
def test_a_custom_check_takes_a_square_root_on_the_numpy_backend():
    scene = alloy.Scene(worlds=2, dt=0.01)
    scene.add_ground()
    scene.add_sphere("ball", 0.05, pos=(0.3, 0.4, 0.05))
    scene.add_robot("cart", mjcf=SLIDE)                                        # something to act on
    reach = C.custom("reach", lambda s: s.xp.sqrt(s["ball.pos"][:, 0] ** 2 + s["ball.pos"][:, 1] ** 2))
    env = VectorEnv(scene, Task(obs=["ball"], reward={"reach": (1.0, reach)}, max_steps=5), worlds=2, device="cpu",
                    backend="numpy")
    try:
        env.reset(seed=0)
        _obs, reward, *_ = env.step(np.zeros((2, 1), np.float32))
        np.testing.assert_allclose(env.xp.to_numpy(reward), 0.5, atol=1e-3)     # 0.3-0.4-0.5 m from the axis
    finally:
        env.close()
