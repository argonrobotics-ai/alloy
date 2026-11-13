# SPDX-License-Identifier: Apache-2.0
"""Without Isaac Lab: every alloy.isaaclab module imports, every name that needs Isaac Lab raises
FeatureUnavailableError with the install line, and ``import alloy`` never loads alloy.isaaclab."""

from __future__ import annotations

import importlib
import importlib.util
import subprocess
import sys

import pytest

from alloy.api.errors import FeatureUnavailableError

ABSENT = importlib.util.find_spec("isaaclab") is None
MODULES = ("alloy.isaaclab", "alloy.isaaclab.register", "alloy.isaaclab.cfg", "alloy.isaaclab.manager",
           "alloy.isaaclab.presets", "alloy.isaaclab.solver", "alloy.isaaclab.health", "alloy.isaaclab.bridge",
           "alloy.isaaclab.compat")


def test_import_alloy_does_not_load_isaaclab_plugin():
    code = "import sys, alloy; print('alloy.isaaclab' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip().splitlines()[-1] == "False"


@pytest.mark.skipif(not ABSENT, reason="Isaac Lab is installed here")
@pytest.mark.parametrize("module", MODULES)
def test_modules_import_without_isaaclab(module):
    importlib.import_module(module)


@pytest.mark.skipif(not ABSENT, reason="Isaac Lab is installed here")
@pytest.mark.parametrize("name", ["AlloySolverCfg", "NewtonAlloyManager", "alloy_physics", "install"])
def test_names_say_how_to_install(name):
    import alloy.isaaclab as il

    with pytest.raises(FeatureUnavailableError) as err:
        obj = getattr(il, name)
        obj(None) if name == "alloy_physics" else obj() if name == "install" else None
    assert "uv pip install isaaclab==3.0.0" in err.value.fix and "alloy-engine" in err.value.fix
