# SPDX-License-Identifier: Apache-2.0
"""Wheeled-vehicle plug-in (Gym-Anything forklift gap): virtual wheels on a chassis body with magic-formula tyres,
rolling resistance, a motor, a gearbox and an open differential.

Example::

    from alloy.plugins.vehicle import Vehicle
    report = alloy.plugins.check(Vehicle(), quick=True)       # crash + laws

It is off by default: nothing changes unless a simulation installs it.
"""

from alloy.plugins.vehicle.plugin import Vehicle

__all__ = ["Vehicle"]
