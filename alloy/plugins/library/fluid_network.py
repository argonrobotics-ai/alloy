# SPDX-License-Identifier: Apache-2.0
"""Lumped hydraulics and pneumatics (engine item 10 part 3): a declarative fluid network with actuators.

``FluidNetwork`` (force) steps a network of the lumped-fluid core (``alloy.plugins.fluid``): chambers (gas or liquid),
edges (lines, orifices, ISO 6358 gas valves, check and relief valves, pumps) and actuators (cylinders on prismatic
joints, rotary vanes on revolute joints) that own their chambers' volumes. Each exchange, ``apply`` reads every actuator's travel from the body poses,
writes its chamber volumes and applies the pressure force on the joint's bodies; ``update`` steps the network
(implicit in the conductances, exact per node volume change). One thread per world for each.

Commands are edge openings in [0, 1] per world (a valve's opening, a pump's command), the input ``opening [W, m]``:
``sim.plugins["fluid_network"].set_opening("supply_a", 1.0)``. Chamber volumes of actuator nodes come from the travel
(the ``Node.volume`` given is replaced at setup; a ``shared`` node holds the sum of its chambers).

Pressure readouts: ``pressure [W, n]`` is each node's at the end of the step. ``piston_pressure [W, A, 2]`` is the gauge
pressure each actuator's force acted with (A side, B side). While oil flows the node reads up to K·Q·dt/V higher: the
force is taken before the step's source inflow is added (docs/known_gaps/fluid-chamber-to-chamber-edges.md).

The default config is a double-acting pneumatic cylinder (20 mm bore, 14 mm rod) on joint ``cyl_slide`` (as built by
``HydraulicCylinder(...).build(builder, xform, "cyl")``) with a 6 bar supply and an exhaust valve per chamber. A plain
dict (``FluidNetwork.from_dict``) declares a network for scene cards and robot files.

Energy (liquid): the core books each edge's throttling loss and each pump's work (``work``) and every chamber's
∫ (P - P_ref)·dV (``node_work``: the work done on the actuators); ``alloy.plugins.fluid.stored_energy`` gives the
compression energy, so pump work = mechanical + throttling + stored (law H-6).

v1 limits: explicit forces (``stable_dt``); gas inflow at the node's temperature (no temperature state); no energy
ledger for gas edges.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import warp as wp

from alloy._vendor import newton

from alloy.plugins.base import ForcePlugin, Forces
from alloy.plugins.fluid import G_TANK, P_ATM, Edge, Network, Node
from alloy.plugins.fluid.actuators import Actuator, Cylinder, Vane, actuate, spec_array, validate
from alloy.plugins.harness.scene import Law, TestScene

SUPPLY = 6.0e5                           # Pa abs, the default circuit's supply
VALVE_C = 1.0e-8                         # m³/(s·Pa), the default valves' ISO 6358 sonic conductance (a fast valve)
DEFAULT_NODES = (Node("cap", volume=1.0e-5, pressure=SUPPLY), Node("rod", volume=1.0e-5, pressure=P_ATM))
DEFAULT_EDGES = (Edge("supply_cap", "orifice_gas", "cap", reservoir=SUPPLY, sonic_conductance=VALVE_C),
                 Edge("exhaust_cap", "orifice_gas", "cap", reservoir=P_ATM, sonic_conductance=VALVE_C),
                 Edge("supply_rod", "orifice_gas", "rod", reservoir=SUPPLY, sonic_conductance=VALVE_C),
                 Edge("exhaust_rod", "orifice_gas", "rod", reservoir=P_ATM, sonic_conductance=VALVE_C))
DEFAULT_ACTUATORS = (Cylinder("cyl", joint="cyl_slide", a="cap", b="rod", bore=0.02, rod=0.02 / math.sqrt(2.0),
                              stroke=0.1, x0=0.05),)
DEFAULT_OPENINGS = (("supply_cap", 1.0), ("exhaust_cap", 0.0), ("supply_rod", 0.0), ("exhaust_rod", 1.0))


def _actuator(a: Any) -> Actuator:
    """A Cylinder or Vane, or one from plain data (a vane has ``displacement``)."""
    if isinstance(a, (Cylinder, Vane)):
        return a
    return Vane(**a) if "displacement" in a else Cylinder(**a)


class FluidNetwork(ForcePlugin):
    """Chambers, valves, lines, pumps and cylinders as one per-world lumped network (pneumatic or hydraulic).

    Example (a single-acting hydraulic lift fed by a pump through a check valve, with a relief valve)::

        net = FluidNetwork(
            nodes=(Node("lift", medium="liquid"),),
            edges=(Edge("pump", "source", "lift", flow=2e-6), Edge("relief", "relief", "lift", conductance=1e-10,
                   p_set=60e5)),
            actuators=(Cylinder("lift", joint="arm_slide", a="lift", bore=0.03, stroke=0.2),))
    """

    name = "fluid_network"
    summary = "lumped hydraulics / pneumatics: chambers, valves, lines, pumps and cylinders on prismatic joints"
    internal = True

    @dataclass
    class Config:
        nodes: tuple[Node, ...] = DEFAULT_NODES
        edges: tuple[Edge, ...] = DEFAULT_EDGES
        actuators: tuple[Actuator, ...] = DEFAULT_ACTUATORS
        openings: tuple[tuple[str, float], ...] = DEFAULT_OPENINGS   # start openings (others 1)
        atmosphere: float = P_ATM

        def __post_init__(self) -> None:
            # plain data (scene cards, dataclasses.asdict round trips) becomes the core's dataclasses
            self.nodes = tuple(n if isinstance(n, Node) else Node(**n) for n in self.nodes)
            self.edges = tuple(e if isinstance(e, Edge) else Edge(**e) for e in self.edges)
            self.actuators = tuple(_actuator(a) for a in self.actuators)
            pairs = self.openings.items() if isinstance(self.openings, dict) else self.openings
            self.openings = tuple((str(k), float(v)) for k, v in pairs)
            net = Network(self.nodes, self.edges)          # raises on a bad network
            validate(tuple(self.actuators), net)
            edges = {e.name for e in self.edges}
            for name, value in self.openings:
                if name not in edges or not 0.0 <= value <= 1.0:
                    raise ValueError(f"openings: {name!r} must be an edge and its value in [0, 1], got {value}")

    @classmethod
    def from_dict(cls, spec: dict[str, Any]) -> FluidNetwork:
        """A network from plain data: {"nodes": [{...Node kwargs}], "edges": [...], "actuators": [{...Cylinder
        kwargs}], "openings": {"edge": value}}."""
        fields = {k: spec[k] for k in ("nodes", "edges", "actuators", "atmosphere") if k in spec}
        return cls(**fields, openings=spec.get("openings", ()))      # no openings given: every edge starts open

    def network(self) -> Network:
        return Network(self.config.nodes, self.config.edges)

    def setup(self, ctx: Any) -> None:
        c, model, W = self.config, ctx.model, ctx.world_count
        net = self.network()
        g = float(np.linalg.norm(ctx.gravity))
        self.fluid = net.compile(ctx.device, W, gravity=g if g > 0.0 else G_TANK)
        acts = tuple(c.actuators)
        self.n_act = len(acts)
        self.joint = ctx.array(np.stack([ctx.joints(a.joint).numpy() for a in acts], axis=1).astype(np.int32)
                               if acts else np.zeros((W, 1), np.int32), wp.int32)
        jt = model.joint_type.numpy()
        for k, a in enumerate(acts):
            want = newton.JointType.REVOLUTE if isinstance(a, Vane) else newton.JointType.PRISMATIC
            if np.any(jt[self.joint.numpy()[:, k]] != int(want)):
                raise ValueError(f"{self.label}: {a.name!r} needs a {want.name.lower()} joint, {a.joint!r} is not")
        self.spec = spec_array(acts, net, ctx.device)
        x_start = self._start_volumes(model, acts, net)
        opening = self.fluid.opening.numpy()
        for name, value in c.openings:
            opening[:, net.edge_index(name)] = value
        self.fluid.opening.assign(opening)
        for k, a in self.fluid.state_arrays().items():
            self.add_state(k, a)
        self.add_state("conductance", self.fluid._g)       # last step's g: the actuators' force pressure reads it
        shape = (W, max(self.n_act, 1))
        self.travel = self.add_state("travel", ctx.array(x_start.astype(np.float32)))
        self.speed = self.add_state("speed", ctx.zeros(shape))
        self.force = self.add_state("force", ctx.zeros(shape))
        self.piston = self.add_state("piston_pressure", ctx.zeros((*shape, 2)))
        self.add_input("opening", self.fluid.opening)

    def _start_volumes(self, model: newton.Model, acts: tuple[Actuator, ...], net: Network) -> np.ndarray:
        """Actuator chambers start at the volumes of their start travel (x0 + the joint's start coordinate); returns
        the start travel [W, max(A, 1)]."""
        q, q_start, vol = model.joint_q.numpy(), model.joint_q_start.numpy(), self.fluid.volume.numpy()
        travel = np.zeros((vol.shape[0], max(len(acts), 1)))
        for a in acts:                                     # a shared node holds the sum of its chambers
            vol[:, [net.node_index(n) for n in (a.a, a.b) if n is not None]] = 0.0
        for k, a in enumerate(acts):
            x = a.x0 + q[q_start[self.joint.numpy()[:, k]]].astype(np.float64)
            travel[:, k] = x
            va, vb = a.volumes(0.0)
            vol[:, net.node_index(a.a)] += np.maximum(va + a.area_a * x, 0.05 * a.dead_a)
            if a.b is not None:
                vol[:, net.node_index(a.b)] += np.maximum(vb - a.area_b * x, 0.05 * a.dead_b)
        self.fluid.volume.assign(vol)
        self.fluid.volume_old.assign(vol)
        return travel

    def set_opening(self, edge: str, values: Any) -> None:
        """Opening of edge ``edge`` (a scalar or one per world, in [0, 1]); host copy, call between periods."""
        a = self.fluid.opening.numpy()
        a[:, self.fluid.net.edge_index(edge)] = np.clip(np.asarray(values, np.float32), 0.0, 1.0)
        self.fluid.opening.assign(a)

    def stable_dt(self, model: newton.Model) -> dict[str, float]:
        """Per actuator, the largest stable exchange step at its start travel, the child body moving alone (a
        cylinder: its mass; a vane: its moment of inertia about the joint axis)."""
        q, q_start, net = model.joint_q.numpy(), model.joint_q_start.numpy(), self.network()
        out = {}
        for k, a in enumerate(self.config.actuators):
            j = int(self.joint.numpy()[0, k])
            out[a.name] = a.check_dt(net, a.x0 + float(q[q_start[j]]), _moving_inertia(model, j, isinstance(a, Vane)))
        return out

    def apply(self, state: newton.State, forces: Forces, dt: float) -> None:
        if self.n_act == 0:
            return
        m, f = self.ctx.model, self.fluid
        wp.launch(actuate, dim=self.ctx.world_count,
                  inputs=[state.body_q, state.body_qd, m.body_com, m.body_inv_mass, m.body_inv_inertia, float(dt),
                          m.joint_parent,
                          m.joint_child, m.joint_X_p, m.joint_X_c, m.joint_axis, m.joint_qd_start, self.n_act,
                          self.joint, self.spec, wp.float64(self.config.atmosphere), f.medium, f.bulk, f.p_ref,
                          f.poly, f.ea, f.eb, f._g, f.pressure, f.volume_old],
                  outputs=[f.volume, self.travel, self.speed, self.force, self.piston, forces.body_f],
                  device=self.ctx.device)

    def update(self, state_in: newton.State, state_out: newton.State, dt: float) -> None:
        self.fluid.step(dt)

    def test_scene(self) -> TestScene:
        from alloy.plugins.library.fluid_network_laws import stall_scene  # noqa: PLC0415

        return stall_scene(self, load=0.9)

    def laws(self) -> list[Law]:
        from alloy.plugins.library.fluid_network_laws import fluid_network_laws  # noqa: PLC0415

        return fluid_network_laws(self)


def _moving_inertia(model: newton.Model, j: int, rotary: bool) -> float:
    """The joint child's mass, or its moment of inertia about the joint axis through the joint anchor."""
    from alloy.plugins.npgeom import quat_rotate  # noqa: PLC0415

    c = int(model.joint_child.numpy()[j])
    if not rotary:
        return float(model.body_mass.numpy()[c])
    xc = model.joint_X_c.numpy()[j].astype(np.float64)
    axis = model.joint_axis.numpy()[int(model.joint_qd_start.numpy()[j])].astype(np.float64)
    a = quat_rotate(xc[3:7], axis / np.linalg.norm(axis))              # axis in the child frame
    r = model.body_com.numpy()[c].astype(np.float64) - xc[:3]
    r_perp = r - (r @ a) * a
    inertia = model.body_inertia.numpy()[c].astype(np.float64)
    return float(a @ inertia @ a + model.body_mass.numpy()[c] * (r_perp @ r_perp))


__all__ = ["DEFAULT_ACTUATORS", "DEFAULT_EDGES", "DEFAULT_NODES", "FluidNetwork", "SUPPLY", "VALVE_C"]
