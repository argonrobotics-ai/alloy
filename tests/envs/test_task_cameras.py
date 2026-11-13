# SPDX-License-Identifier: Apache-2.0
"""``Task(cameras=[cam])`` raises Alloy's error with the fix, not a bare TypeError (audit T2-004)."""

from __future__ import annotations

import pytest

import alloy
from alloy.api.errors import AlloyAPIError
from alloy.envs import EnvError, Task, VectorEnv


@pytest.mark.parametrize("cameras", [[object()], ("front",), {1: object()}])
def test_cameras_not_a_dict_of_names_raise_with_the_fix(cameras):
    with pytest.raises(EnvError) as e:
        Task(cameras=cameras).validate()
    assert isinstance(e.value, AlloyAPIError) and e.value.field == "cameras" and "cameras={" in e.value.fix
    scene = alloy.Scene(dt=0.02)
    scene.add_ground()
    with pytest.raises(EnvError):                                        # before anything is built
        VectorEnv(scene, Task(cameras=cameras), worlds=1, device="cpu", backend="numpy")
