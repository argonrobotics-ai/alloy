# SPDX-License-Identifier: Apache-2.0
"""What the envs say when a package is missing (ease test, 2026-11-09), and check arithmetic (CPU, no simulation).

A fresh install has neither PyTorch nor huggingface_hub: the torch backend and PickPlace-v0's Hub object must name
the command that installs them (the ``envs`` extra once the package declares it), not a list of unrelated cards.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from alloy.envs import checks as C
from alloy.envs import errors
from alloy.envs.backend import NumpyBackend, pick_backend
from alloy.envs.errors import EnvError


HUB = "pip install 'alloy-engine[import]'"


@pytest.mark.parametrize(("extra", "torch", "hub"), [(False, "pip install torch", HUB),
                                                     (True, "alloy-engine[envs]", "alloy-engine[envs]")])
def test_install_fix_names_the_envs_extra_once_it_exists(monkeypatch, extra: bool, torch: str, hub: str) -> None:
    monkeypatch.setattr(errors, "has_extra", lambda name: extra and name == "envs")
    assert torch in errors.install_fix("torch")
    assert hub in errors.install_fix("huggingface_hub")
    assert "install.sh" not in errors.install_fix("torch") + errors.install_fix("huggingface_hub")   # pip only


def test_the_torch_backend_names_the_install(monkeypatch) -> None:
    from alloy.envs import backend  # noqa: PLC0415

    monkeypatch.setattr(backend, "has_torch", lambda: False)
    monkeypatch.setattr(errors, "has_extra", lambda name: False)
    with pytest.raises(EnvError) as e:
        pick_backend("torch", "cpu")
    assert "pip install torch" in e.value.fix and "backend='numpy'" in e.value.fix
    assert isinstance(pick_backend("auto", "cpu"), NumpyBackend)          # auto falls back to NumPy


def test_pick_place_names_the_hub_install_not_other_cards(monkeypatch) -> None:
    from alloy.api import catalog  # noqa: PLC0415

    from alloy.envs.library_tasks import pick_place_scene  # noqa: PLC0415

    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a: None if name == "huggingface_hub" else real(name))
    monkeypatch.setattr(catalog, "can_place", lambda name, extra=(): False)
    monkeypatch.setattr(errors, "has_extra", lambda name: False)
    with pytest.raises(EnvError) as e:
        pick_place_scene(1, vary=False)
    assert "huggingface_hub is not installed" in e.value.message and "PickPlace-v0" in e.value.message
    assert "alloy-engine[import]" in e.value.fix and "install.sh" not in e.value.fix


class _State:
    xp = NumpyBackend("cpu")


def test_checks_do_arithmetic() -> None:
    a = C.custom("a", lambda s: np.array([0.2, 0.5]))
    b = C.custom("b", lambda s: np.array([True, False]))           # a bool check counts 1 / 0
    s = _State()
    np.testing.assert_allclose((1.0 - a - b)(s), [-0.2, 0.5])
    np.testing.assert_allclose((2 * a + b / 2)(s), [0.9, 1.0])
    np.testing.assert_allclose((-a)(s), [-0.2, -0.5])
    assert (1.0 - a).name == "(1 - a)" and ((a * b) > 0.1)(s).tolist() == [True, False]
