# SPDX-License-Identifier: Apache-2.0
"""Alloy: a multi-material, parallel physics engine for robot learning, built on Newton and MuJoCo Warp.

``import alloy`` is light: it loads only the standard library. The public names (``alloy.Scene``,
``alloy.load``, ``alloy.build``, ...; ``alloy.api``) import their module on first use, and engine set-up that needs
Newton runs when Newton is imported.
"""

__version__ = "0.1.2"


def _installed_version(fallback: str, package_dir: str | None = None) -> str:
    """The version pip installed: the one ``alloy_engine-<version>.dist-info`` folder beside this package (a
    post-release such as 0.1.0.post1 differs from the source line above); ``fallback`` in a checkout, an editable
    install or when there is not exactly one. One directory listing, standard library only; never raises.
    Example: ``_installed_version("0.1.1", "/site-packages/alloy")`` -> ``"0.1.0.post1"``."""
    import os

    try:
        site = os.path.dirname(os.path.abspath(package_dir or os.path.dirname(__file__)))
        found = [e.name[len("alloy_engine-"):-len(".dist-info")] for e in os.scandir(site)
                 if e.name.startswith("alloy_engine-") and e.name.endswith(".dist-info")]
    except Exception:  # noqa: BLE001 - a version lookup must never stop `import alloy`
        return fallback
    return found[0] if len(found) == 1 and found[0] else fallback


__version__ = _installed_version(__version__)

import sys as _sys

from alloy.api._hooks import after_import as _after_import
from alloy.api._lazy import PUBLIC as __all__  # noqa: F401
from alloy.api._lazy import module_hooks as _module_hooks


# Compensated float32 joint coordinates on every SolverMuJoCo built in this process (SCR-SC10-3, coordinator ruling
# 2026-11-05 22:33Z; WP3, alloy.scene.coordinate_carry). Opt out per entry with {"alloy:coordinate_carry": False}.
# Installed when Newton is imported (no SolverMuJoCo exists before that).
def _install_coordinate_carry() -> None:
    carry = _sys.modules.get("alloy.scene.coordinate_carry")
    if carry is not None and not hasattr(carry, "install_default"):
        return  # its own `import newton` fired this hook mid-import; the module installs the default at its end
    from alloy.scene.coordinate_carry import install_default

    install_default()


_after_import("alloy._vendor.newton", _install_coordinate_carry)


# MuJoCo Warp's collision glue calls the @wp.func upper_trid_index from Python (~40 us of overload resolution each, 81
# calls a T12 step); a plain-Python twin is rebound in those two modules as they load (SCR-WP8-5, alloy.core.mjw_fastpath).
def _mjw_fastpath(module_name: str):  # noqa: ANN202
    def install() -> None:
        from alloy.core.mjw_fastpath import install as _install

        _install(module_name)

    return install


for _name in ("alloy._vendor.mujoco_warp._src.collision_convex", "alloy._vendor.mujoco_warp._src.collision_primitive"):
    _after_import(_name, _mjw_fastpath(_name))
del _name

# MuJoCo Warp's EPA with mujoco_warp PR #1845's first six commits, vendored until our pin has the fix (WP7's SB-7
# EPA_HORIZON S1s; WP3, alloy.contact.mjw_epa): the stock horizon overflows at ccd_iterations 35 and drops that world's
# contacts for a step. Standard library only here: it patches MuJoCo Warp's EPA users when they are imported.
from alloy.contact.mjw_epa import install as _install_mjw_epa  # noqa: E402

_install_mjw_epa()

__getattr__, __dir__ = _module_hooks(globals(), "alloy")
