# SPDX-License-Identifier: Apache-2.0
"""Host side of the lumped-fluid core: declare chambers and their connections, compile them to per-world arrays.

Example (one vacuum cup chamber)::

    net = Network([Node("cup", volume=5e-6)],
                  [Edge("pump", "pump_gas", "cup", reservoir=16e3, conductance=1e-4),
                   Edge("leak", "linear", "cup", conductance=1e-6)])
    fluid = net.compile(device, world_count)          # FluidState: pressure [W, n], ledger [W, m], ...
    fluid.step(dt)                                    # launch from a plug-in hook (graph-safe)

Units: SI, absolute pressures. A gas network carries throughput [Pa·m³/s] (conductances in m³/s, vacuum
convention); a liquid network carries volume flow [m³/s] (conductances in m³/(s·Pa)). An edge never joins a gas and a
liquid node, except through a ``port`` between two-phase ``mixed`` chambers (``alloy.plugins.fluid.mixed``: liquid
with a gas pocket, a syringe or a vial; such a network holds only mixed nodes and ports).
An open tank (``medium="tank"``) is a liquid node under a free surface, P = P_ref + ρ·g·V/A: sink
compartments, buckets, a drained film's catch basin (``Node("sink", medium="tank", volume=0.02, area=0.16)``). See ``alloy/plugins/fluid/kernels.py`` for the integration.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import warp as wp

from alloy.plugins.fluid import kernels as K

P_ATM = 101325.0                                          # Pa
MEDIA = {"gas": 0, "liquid": 1, "tank": 2, "mixed": 3}
G_TANK = 9.81                                             # m/s², the default head of an open tank (compile(gravity=))
KINDS = {"linear": 0, "orifice": 1, "check": 2, "relief": 3, "pump_gas": 4, "source": 5, "orifice_gas": 6,
         "expel": 7, "overflow": 3, "port": 8}          # overflow: a relief at a tank's rim, set at compile
RESERVOIR_MEDIA = ("gas", "liquid")                     # what a port's reservoir passes into a mixed node
TO_RESERVOIR = ("relief", "pump_gas", "expel", "overflow")                     # these always end at a reservoir


def _species(values: object, who: str) -> tuple[float, ...]:
    out = tuple(float(v) for v in values)                   # plain data (lists) becomes a tuple
    if any(not v >= 0.0 for v in out):
        raise ValueError(f"{who}: species concentrations must be >= 0, got {out}")
    return out


@dataclass(frozen=True)
class Node:
    """A chamber. ``volume`` is the start volume (the owner may write ``FluidState.volume`` every step).

    An open tank (``medium="tank"``) holds ``volume`` m³ of liquid at the start under a free surface of ``area`` m²
    (prismatic) at ``p_ref`` (the air above it): its outlet pressure is P_ref + ρ·g·V/A, ``density`` ρ; ``pressure``
    is unused. Its volume is the amount (``FluidState.amount``), never written by an owner."""

    name: str
    medium: str = "gas"                  # "gas" (ideal gas, index ``polytropic``) or "liquid" (bulk modulus)
    volume: float = 1.0e-5               # m³
    pressure: float = P_ATM              # Pa, absolute, at the start
    bulk_modulus: float = 1.5e9          # Pa (liquid)
    p_ref: float = P_ATM                 # Pa (liquid: the fluid fills ``volume`` at this pressure)
    polytropic: float = 1.0              # gas: n of P·V^n on a volume change, κ = V/n (1 isothermal, 1.4 adiabatic
                                         # air); inflow arrives at the node's temperature (no temperature state)
    area: float = 0.0                    # m², tank: free-surface area
    density: float = 998.0               # kg/m³, tank: the liquid's density (its head ρ·g·V/A)
    species: tuple[float, ...] = ()      # kg/m³, liquid and tank: start concentration of each dissolved species
    capacity: float = math.inf           # m³, tank: the volume to its rim (``tank_overflow`` spills above it)
    liquid: float = 0.0                  # m³ at p_ref, mixed: the start liquid (the rest of ``volume`` is the pocket)
    radius: float = 0.0                  # m, mixed: the chamber's bore radius (its ports' level geometry)
    up: float = 1.0                      # mixed: start cos(head axis, up) (owner-written per world after)
    shared: bool = False                 # a common line: the chamber of several actuators, its volume their sum
                                         # (the line's own oil goes in their dead volumes)

    def __post_init__(self) -> None:
        object.__setattr__(self, "species", _species(self.species, f"node {self.name!r}"))
        if self.medium not in MEDIA:
            raise ValueError(f"node {self.name!r}: medium must be one of {sorted(MEDIA)}, got {self.medium!r}")
        if self.medium == "mixed" and (self.radius <= 0 or self.liquid < 0 or self.pressure <= 0 or
                                       self.volume <= 0 or not -1.0 <= self.up <= 1.0):
            raise ValueError(f"mixed node {self.name!r}: radius > 0, liquid >= 0, pressure > 0, volume > 0 and up in "
                             "[-1, 1] are required")
        if self.medium == "tank":
            if self.volume < 0 or self.area <= 0 or self.density <= 0:
                raise ValueError(f"tank {self.name!r}: volume >= 0, area > 0 and density > 0 are required")
            return
        if self.volume <= 0 or self.bulk_modulus <= 0:
            raise ValueError(f"node {self.name!r}: volume and bulk_modulus must be > 0")
        if self.medium == "gas" and self.pressure <= 0:
            raise ValueError(f"node {self.name!r}: a gas node needs an absolute pressure > 0")
        if self.polytropic < 1.0 or (self.medium == "liquid" and self.polytropic != 1.0):
            raise ValueError(f"node {self.name!r}: polytropic must be >= 1 (gas only), got {self.polytropic}")


@dataclass(frozen=True)
class Edge:
    """A connection from node ``a`` to node ``b``, or to a reservoir at ``reservoir`` Pa when ``b`` is None.

    Flow Q = g·(P_b - P_a) into ``a``. ``kind``: linear (g = conductance), orifice (liquid; Cd·A = ``cd_area``,
    linearised below ``dp_lam``), check (open while P_b - P_a > ``p_set``), relief (vents G·(P_a - ``p_set``) while
    P_a > ``p_set``; ``reservoir`` is unused), pump_gas (a vacuum pump of speed ``conductance`` down to ``reservoir`` = its ultimate pressure),
    source (``flow`` into ``a``), orifice_gas (ISO 6358: throughput C·p_up·ψ(p_down/p_up)·ρ0·R·T0, sonic conductance
    C = ``sonic_conductance`` [m³/(s·Pa)], ψ = 1 when choked at p_down/p_up <= ``critical_ratio``; linearised below
    ``dp_lam``), overflow (a relief at tank ``a``'s rim, P_ref + ρ·g·capacity/A at the scene's gravity; see
    ``tank_overflow``), expel (an aerosol can's valve: liquid of ``density`` leaves gas node ``a``'s container at
    ``cd_area``·sqrt(2·(P_a - ``reservoir``)/ρ) until ``capacity`` m³ are gone, and the gas expands into the freed
    volume, so the discharge falls as the can empties; its ledger and discharge are that liquid volume). Every edge has a per-world opening in [0, 1] (``FluidState.opening``, default 1)
    that scales its g or flow.
    """

    name: str
    kind: str
    a: str
    b: str | None = None
    reservoir: float = P_ATM             # Pa, when b is None
    conductance: float = 0.0             # m³/s (gas) or m³/(s·Pa) (liquid)
    cd_area: float = 0.0                 # m², orifice
    density: float = 1000.0              # kg/m³, orifice
    dp_lam: float = 1.0e3                # Pa, orifice
    p_set: float = 0.0                   # Pa: check cracking ΔP, relief set pressure (absolute)
    flow: float = 0.0                    # m³/s or Pa·m³/s, source
    sonic_conductance: float = 0.0       # m³/(s·Pa), orifice_gas (ISO 6358 C)
    critical_ratio: float = 0.528        # orifice_gas (ISO 6358 b): choked at p_down/p_up <= b
    species: tuple[float, ...] = ()      # kg/m³ of each species in what a reservoir or source brings in (liquid)
    capacity: float = 0.0                # m³, expel: the liquid in the can
    gas_conductance: float = 0.0         # m³/(s·Pa), port: laminar gas law (``conductance`` is the liquid's)
    depth_a: float = 0.0                 # m, port: depth of the port in node a below its head (on the axis)
    depth_b: float = 0.0                 # m, port: the same in node b
    reservoir_medium: str = "gas"        # port to a reservoir: what flows in from it ("gas" or "liquid")

    def __post_init__(self) -> None:
        object.__setattr__(self, "species", _species(self.species, f"edge {self.name!r}"))
        if self.kind not in KINDS:
            raise ValueError(f"edge {self.name!r}: kind must be one of {sorted(KINDS)}, got {self.kind!r}")
        if self.reservoir_medium not in RESERVOIR_MEDIA or min(self.gas_conductance, self.depth_a, self.depth_b) < 0:
            raise ValueError(f"edge {self.name!r}: reservoir_medium in {RESERVOIR_MEDIA}, gas_conductance and depths "
                             ">= 0 are required")
        if self.kind in (*TO_RESERVOIR, "source") and self.b is not None:
            raise ValueError(f"edge {self.name!r}: a {self.kind} edge ends at a reservoir (b=None)")
        if min(self.conductance, self.cd_area, self.dp_lam, self.sonic_conductance) < 0 or self.density <= 0:
            raise ValueError(f"edge {self.name!r}: conductances, cd_area, dp_lam >= 0 and density > 0 are required")
        if not 0.0 <= self.critical_ratio < 1.0:
            raise ValueError(f"edge {self.name!r}: critical_ratio must be in [0, 1), got {self.critical_ratio}")


class Network:
    """Nodes and edges of one world's network (every world gets a copy). Up to 8 nodes and 16 edges."""

    def __init__(self, nodes: Sequence[Node], edges: Sequence[Edge]) -> None:
        self.nodes, self.edges = tuple(nodes), tuple(edges)
        names = [n.name for n in self.nodes]
        if not 1 <= len(names) <= K.MAX_NODES or len(set(names)) != len(names):
            raise ValueError(f"1-{K.MAX_NODES} uniquely named nodes are required, got {names}")
        if len(self.edges) > K.MAX_EDGES or len({e.name for e in self.edges}) != len(self.edges):
            raise ValueError(f"at most {K.MAX_EDGES} uniquely named edges are allowed")
        self.mixed = any(n.medium == "mixed" for n in self.nodes) or any(e.kind == "port" for e in self.edges)
        if self.mixed:
            self._check_mixed(names)
            return
        medium = {n.name: "gas" if n.medium == "gas" else "liquid" for n in self.nodes}   # a tank is liquid
        for e in self.edges:
            for end in (e.a, e.b):
                if end is not None and end not in medium:
                    raise ValueError(f"edge {e.name!r}: unknown node {end!r} (nodes: {names})")
            if e.b is not None and medium[e.a] != medium[e.b]:
                raise ValueError(f"edge {e.name!r} joins a {medium[e.a]} and a {medium[e.b]} node: their flows have "
                                 "different units (throughput vs volume); use one medium per network")
            if e.kind == "orifice" and medium[e.a] != "liquid":
                raise ValueError(f"edge {e.name!r}: the orifice edge is the liquid law (gas: orifice_gas)")
            if e.kind == "orifice_gas" and medium[e.a] != "gas":
                raise ValueError(f"edge {e.name!r}: orifice_gas is the gas law (liquid: orifice)")
            tanks = {n.name: n for n in self.nodes if n.medium == "tank"}
            tank = tanks.get(e.a)
            if tank is not None and e.b is None and e.kind not in ("source", "relief", "overflow") and \
                    e.reservoir < tank.p_ref:
                raise ValueError(f"edge {e.name!r}: tank {e.a!r} drains to {e.reservoir} Pa, below its surface pressure "
                                 f"{tank.p_ref} Pa: it would run below empty (lumped tanks hold no air)")
            if e.kind == "overflow" and (e.a not in tanks or not math.isfinite(tanks[e.a].capacity)):
                raise ValueError(f"edge {e.name!r}: an overflow needs a tank with a finite capacity (its rim)")
            if e.kind == "expel" and (medium[e.a] != "gas" or e.cd_area <= 0 or e.capacity < 0):
                raise ValueError(f"edge {e.name!r}: expel drains a gas node's can (cd_area > 0, capacity >= 0)")
            if e.kind == "pump_gas" and medium[e.a] != "gas":
                raise ValueError(f"edge {e.name!r}: pump_gas pumps a gas node (a liquid pump is a source edge)")

    def _check_mixed(self, names: list[str]) -> None:
        """A network with a mixed node: only mixed nodes, only ports between them or to reservoirs, no species."""
        plain = [n.name for n in self.nodes if n.medium != "mixed"]
        if plain:
            raise ValueError(f"nodes {plain}: a network with a mixed node (or a port) holds only mixed nodes; a mixed "
                             "port never joins a gas, liquid or tank node")
        for e in self.edges:
            if e.kind != "port":
                raise ValueError(f"edge {e.name!r}: a mixed network's edges are ports, got {e.kind!r}")
            for end in (e.a, e.b):
                if end is not None and end not in names:
                    raise ValueError(f"edge {e.name!r}: unknown node {end!r} (nodes: {names})")
        if self.species:
            raise ValueError("a mixed network carries no dissolved species (v1)")

    @property
    def species(self) -> int:
        """The number of dissolved species: the longest ``species`` tuple of any node or edge (missing = 0)."""
        return max([len(x.species) for x in (*self.nodes, *self.edges)] or [0])

    def tank_kappa(self, gravity: float = G_TANK) -> dict[str, float]:
        """κ = A/(ρ·g) [m³/Pa] of each tank node, at ``gravity`` [m/s²] (the scene's |g|)."""
        return {nd.name: nd.area / (nd.density * gravity) for nd in self.nodes if nd.medium == "tank"}

    def node_index(self, name: str) -> int:
        return [n.name for n in self.nodes].index(name)

    def edge_index(self, name: str) -> int:
        return [e.name for e in self.edges].index(name)

    def compile(self, device: object, world_count: int, gravity: float = G_TANK) -> FluidState:
        """Per-world device arrays; ``gravity`` [m/s²] sets the open tanks' heads (pass the scene's |g|)."""
        return FluidState(self, device, world_count, gravity)


class FluidState:
    """A compiled network: per-world device arrays and the step launch. Arrays (W = worlds, n nodes, m edges):
    ``pressure [W, n]`` float64 Pa, ``volume [W, n]`` float64 m³ (owner-written), ``volume_old [W, n]``,
    ``opening [W, m]`` float32 in [0, 1], ``ledger [W, m]`` float64 amount carried into node ``a`` since the start,
    ``discharge [W, m]`` the last step's mean flow into node ``a`` (ledger increment / dt), ``work [W, m]`` (liquid
    edges: throttling loss, or a source's pump work) and ``node_work [W, n]`` (∫ (P - P_ref)·dV) in J;
    ``inflow [W, n]`` float64 an external inflow over the next step (amount/s; e.g. a film's drains into a tank),
    consumed by ``step`` and booked in ``inflow_total [W, n]``.

    With ``K = net.species > 0`` dissolved species ride the liquid (``kernels.step_species``): ``species [W, n, K]``
    the mass in each node [kg], ``species_ledger [W, m, K]`` the mass each edge carried into node ``a``,
    ``inflow_species [W, n, K]`` an external species inflow [kg/s] over the next step (consumed, booked in
    ``species_in_total``); ``concentration()`` gives kg/m³; ``out_conc [W, n, K]`` the concentration every outflow of
    a node carried in the last step [kg/m³] (also a node that step emptied, whose M/amount reads 0)."""

    def __init__(self, net: Network, device: object, world_count: int, gravity: float = G_TANK) -> None:
        self.net, self.device, self.world_count, self.gravity = net, device, int(world_count), float(gravity)
        n, m, W = len(net.nodes), len(net.edges), self.world_count
        self.n, self.m = n, m
        idx = {nd.name: i for i, nd in enumerate(net.nodes)}

        def arr(values: object, dtype: object) -> wp.array:
            return wp.array(np.asarray(values), dtype=dtype, device=device)

        self.medium = arr([MEDIA[nd.medium] for nd in net.nodes], wp.int32)
        tanks = net.tank_kappa(self.gravity)
        self.bulk = arr([tanks.get(nd.name, nd.bulk_modulus) for nd in net.nodes], wp.float64)   # tank: κ
        self.p_ref = arr([nd.p_ref for nd in net.nodes], wp.float64)
        self.poly = arr([nd.polytropic for nd in net.nodes], wp.float64)
        es = net.edges or (Edge("_none", "linear", net.nodes[0].name),)      # kernels want non-empty arrays
        self.kind = arr([KINDS[e.kind] for e in es], wp.int32)
        self.ea = arr([idx[e.a] for e in es], wp.int32)
        self.eb = arr([-1 if e.b is None else idx[e.b] for e in es], wp.int32)
        # a relief valve vents G·(P - p_set) above its set pressure: its flow law references p_set (the vented
        # amount leaves the network, wherever the tank is)
        nodes = {nd.name: nd for nd in net.nodes}

        def rim(e: Edge) -> float:                       # an overflow vents above the tank's rim, at this gravity
            nd = nodes[e.a]
            return nd.p_ref + nd.density * self.gravity * nd.capacity / nd.area

        self.p_res = arr([rim(e) if e.kind == "overflow" else e.p_set if e.kind == "relief" else e.reservoir
                          for e in es], wp.float64)
        self.g0 = arr([e.conductance for e in es], wp.float64)
        gas_orifice = [e.kind == "orifice_gas" for e in es]                     # C and b ride in cd_area / p_set
        self.cd_area = arr([e.sonic_conductance if o else e.cd_area for e, o in zip(es, gas_orifice, strict=True)],
                           wp.float64)
        self.rho = arr([e.density for e in es], wp.float64)
        self.dp_lam = arr([e.dp_lam for e in es], wp.float64)
        self.p_set = arr([e.critical_ratio if o else rim(e) if e.kind == "overflow" else e.p_set
                          for e, o in zip(es, gas_orifice, strict=True)], wp.float64)
        self.flow = arr([e.capacity if e.kind == "expel" else e.flow for e in es], wp.float64)   # expel: capacity
        me = max(m, 1)
        p0 = np.tile([nd.p_ref + nd.volume / tanks[nd.name] if nd.name in tanks else nd.pressure
                      for nd in net.nodes], (W, 1))
        v0 = np.tile([nd.volume for nd in net.nodes], (W, 1))
        self.pressure = arr(p0, wp.float64)
        self.volume = arr(v0, wp.float64)
        self.volume_old = arr(v0, wp.float64)
        self.opening = arr(np.ones((W, me), np.float32), wp.float32)
        self.ledger = arr(np.zeros((W, me)), wp.float64)
        self.work = arr(np.zeros((W, me)), wp.float64)
        self.node_work = arr(np.zeros((W, n)), wp.float64)
        self.discharge = arr(np.zeros((W, me)), wp.float64)
        self.inflow = arr(np.zeros((W, n)), wp.float64)
        self.inflow_total = arr(np.zeros((W, n)), wp.float64)
        self.n_species = K = net.species
        ks = max(K, 1)
        pad = lambda v: list(v) + [0.0] * (ks - len(v))  # noqa: E731
        self.conc_in = arr([pad(e.species) for e in es], wp.float64)
        a0 = np.zeros_like(p0) if net.mixed else self.amount(p0, v0)
        c0 = np.asarray([pad(nd.species) for nd in net.nodes])                    # [n, K] kg/m³
        mass = a0[..., None] * c0[None]
        gas = [i for i, nd in enumerate(net.nodes) if nd.medium == "gas"]
        mass[:, gas, :] = 0.0
        self.species = arr(mass, wp.float64)
        conc = np.broadcast_to(c0[None], mass.shape).copy()
        conc[:, gas, :] = 0.0
        self.out_conc = arr(conc, wp.float64)            # [W, n, K] kg/m³: what each node's outflows carried last step
        self.species_ledger = arr(np.zeros((W, me, ks)), wp.float64)
        self.inflow_species = arr(np.zeros((W, n, ks)), wp.float64)
        self.species_in_total = arr(np.zeros((W, n, ks)), wp.float64)
        self._g = arr(np.zeros((W, me)), wp.float64)
        self._mat = arr(np.zeros((W, n, n)), wp.float64)
        self._rhs = arr(np.zeros((W, n)), wp.float64)
        self._pv = arr(np.zeros((W, n)), wp.float64)
        self.mixed = None
        if net.mixed:                                    # liquid + gas pocket chambers: their own step
            from alloy.plugins.fluid.mixed import MixedState  # noqa: PLC0415 - mixed imports this module

            self.mixed = MixedState(self)

    def state_arrays(self) -> dict[str, wp.array]:
        """What the next step reads: register these with ``Plugin.add_state`` (snapshots, resets, rollback)."""
        out = {"pressure": self.pressure, "volume": self.volume, "volume_old": self.volume_old,
               "ledger": self.ledger, "work": self.work, "node_work": self.node_work, "discharge": self.discharge,
               "inflow_total": self.inflow_total}
        if self.n_species:
            out.update(species=self.species, species_ledger=self.species_ledger,
                       species_in_total=self.species_in_total, out_conc=self.out_conc)
        if self.mixed is not None:
            out.update(self.mixed.state_arrays())
        return out

    def step(self, dt: float) -> None:
        """Advance every world's network by ``dt`` (one kernel launch; graph-capturable)."""
        if self.mixed is not None:
            self.mixed.step(dt)
            return
        wp.launch(K.step_network, dim=self.world_count,
                  inputs=[self.n, self.m, wp.float64(dt), self.medium, self.bulk, self.p_ref, self.poly, self.kind,
                          self.ea, self.eb, self.p_res, self.g0, self.cd_area, self.rho, self.dp_lam, self.p_set,
                          self.flow, self.opening, self.volume, self.inflow],
                  outputs=[self._g, self._mat, self._rhs, self._pv, self.pressure, self.volume_old, self.ledger, self.work,
                           self.node_work, self.discharge, self.inflow_total],
                  device=self.device)
        if self.n_species:
            wp.launch(K.step_species, dim=self.world_count,
                      inputs=[self.n, self.m, self.n_species, wp.float64(dt), self.medium, self.bulk, self.p_ref,
                              self.poly, self.ea, self.eb, self.conc_in, self.discharge, self.pressure, self.volume],
                      outputs=[self._mat, self._rhs, self.species, self.species_ledger, self.inflow_species,
                               self.species_in_total, self.out_conc], device=self.device)

    def set_opening(self, edge: str, values: wp.array) -> None:
        """opening[:, edge] = values [W] (float32), in a kernel (graph-safe)."""
        wp.launch(K.set_opening, dim=self.world_count, inputs=[self.net.edge_index(edge), values],
                  outputs=[self.opening], device=self.device)

    def concentration(self) -> np.ndarray:
        """Host: each node's species concentration [W, n, K] in kg/m³ (0 for gas and empty nodes)."""
        a = self.amount(self.pressure.numpy(), self.volume.numpy())[..., None]
        m = self.species.numpy()[..., :self.n_species]
        return np.divide(m, a, out=np.zeros_like(m), where=a > 0)

    def amount(self, pressure: np.ndarray, volume: np.ndarray) -> np.ndarray:
        """Host: each node's amount κ(V)·P + λ(V) (gas P·V/n Pa·m³, liquid m³; tank: its liquid volume
        κ·(P - P_ref) m³), for ledgers and laws. A mixed network's amounts are ``mixed.liquid`` and ``mixed.gas``."""
        if self.net.mixed:
            raise ValueError("a mixed network has two amounts per node: read FluidState.mixed.liquid and .gas")
        out = np.array(pressure, np.float64) * np.array(volume, np.float64)
        tanks = self.net.tank_kappa(self.gravity)
        for i, nd in enumerate(self.net.nodes):
            if nd.medium == "gas" and nd.polytropic != 1.0:
                out[..., i] /= nd.polytropic
        for i, nd in enumerate(self.net.nodes):
            if nd.medium == "liquid":
                v = np.asarray(volume)[..., i]
                out[..., i] = v / nd.bulk_modulus * np.asarray(pressure)[..., i] + v * (1.0 - nd.p_ref / nd.bulk_modulus)
            elif nd.medium == "tank":
                out[..., i] = tanks[nd.name] * (np.asarray(pressure, np.float64)[..., i] - nd.p_ref)
        return out


def stored_energy(net: Network, pressure: np.ndarray, volume: np.ndarray, gravity: float = G_TANK) -> np.ndarray:
    """Host: each liquid node's compression energy V·(P - P_ref)²/(2K) [J], each tank's potential energy above its
    outlet ρ·g·V²/(2A) = κ·(P - P_ref)²/2 (gas nodes 0), ``[..., n]``."""
    if net.mixed:
        raise ValueError("stored_energy: a mixed network's pockets store exergy, not this; see the syringe laws (G-6)")
    out = np.zeros(np.broadcast(np.asarray(pressure), np.asarray(volume)).shape)
    tanks = net.tank_kappa(gravity)
    for i, nd in enumerate(net.nodes):
        dp = np.asarray(pressure, np.float64)[..., i] - nd.p_ref
        if nd.medium == "liquid":
            out[..., i] = np.asarray(volume, np.float64)[..., i] * dp * dp / (2.0 * nd.bulk_modulus)
        elif nd.medium == "tank":
            out[..., i] = 0.5 * tanks[nd.name] * dp * dp
    return out


def tank_overflow(tank: Node, conductance: float = 1.0e-5, name: str | None = None) -> Edge:
    """An overflow for open tank ``tank`` at its rim: an ``overflow`` edge (a relief) venting G·(P - P_rim) once the
    liquid stands above ``tank.capacity``; P_rim = P_ref + ρ·g·capacity/A at the scene's gravity (set at compile).
    The default G carries about 1 L/s at 10 mm over the rim; the spill is the edge's ledger.
    Example: ``Network([sink], [..., tank_overflow(sink)])``; as data: ``{"kind": "overflow", "a": "sink", ...}``."""
    if tank.medium != "tank" or not math.isfinite(tank.capacity):
        raise ValueError(f"tank_overflow: {tank.name!r} must be a tank with a finite capacity")
    return Edge(name or f"{tank.name}_overflow", "overflow", tank.name, conductance=conductance)


def laminar_conductance(diameter: float, length: float, viscosity: float) -> float:
    """Poiseuille conductance of a round line, π·d⁴/(128·μ·L) [m³/(s·Pa)]: a liquid ``linear`` edge's ``conductance``.
    Example: ``Edge("line", "linear", "a", "b", conductance=laminar_conductance(4e-3, 2.0, 0.03))``."""
    if min(diameter, length, viscosity) <= 0:
        raise ValueError("diameter, length and viscosity must be > 0")
    return float(np.pi * diameter ** 4 / (128.0 * viscosity * length))


__all__ = ["G_TANK", "Edge", "FluidState", "KINDS", "MEDIA", "Network", "Node", "P_ATM", "RESERVOIR_MEDIA",
           "laminar_conductance", "stored_energy", "tank_overflow"]
