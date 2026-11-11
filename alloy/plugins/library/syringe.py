# SPDX-License-Identifier: Apache-2.0
"""Syringes, vials and needles on the lumped-fluid core's two-phase chambers (SCR-WP8-11 Part A).

A syringe barrel and a vial are ``mixed`` nodes (``alloy.plugins.fluid.mixed``: liquid with a gas pocket on top, at
one pressure); needles are ``port`` edges between them, or to a reservoir (the air, a vein, tissue). ``Syringes``
(force) reads each plunger's travel from its prismatic joint every exchange (any solver), writes the barrel volume,
sets which way each chamber's head points (the barrel from its joint axis, a vial from its body or a fixed ``up``) and
applies the pressure on the plunger and back on the barrel; ``update`` steps the network.

    Syringe (prismatic ``joint``: the barrel the parent, the plunger the child, the axis from the head toward the
    thumb rest):   x = x0 + q (the stopper's travel from the head),  V = dead + A·x,
    F along the axis on the plunger = (P - P_atm)·A - push - friction        (``push`` [W, S] N: a thumb, input)

Readings: ``fluid.mixed`` (``liquid``, ``gas``, ``gas_ledger``: GA's air in a barrel or into a vein, in mL via
``air_ml`` and ``air_out_ml``), ``travel``, ``speed``, ``force``.

Stability (WP13's review): the force is explicit, so the plunger joint carries the needle's liquid column as armature,
ρ·L·A²/a (``inertance``; real physics, the column's added mass). Then the needle's damper A²/G needs dt < 2·I·G/A²
(ρ·r²/(4·μ): 4.2 ms for 25 G) and a capped all-liquid barrel, a spring A²·K/V, needs dt < 2·sqrt(I·V/(A²·K)).
``check_dt`` refuses a step past either and says how much armature it needs. v1 limit: the column's mass rides along
when only air moves or the syringe is capped, so a plunger pushed by a bare constant force gathers momentum while the
air leaves, rams the liquid and draws air back in (docs/known_gaps/syringe-force-push-air-phase.md); a finger moving
it at a set speed (a joint drive, a robot) is not affected.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import warp as wp

from alloy._vendor import newton

from alloy.plugins.base import ForcePlugin, Forces
from alloy.plugins.errors import PluginError
from alloy.plugins.fluid import P_ATM, Edge, Network, Node
from alloy.plugins.fluid.mixed import MU_AIR, P_VAPOUR, MixedNodes, MixedPorts, capacitance, cavity, closure
from alloy.plugins.funcs import add_wrench, com_world, point_velocity, wrench_at

WATER_K = 2.2e9                          # Pa, water's bulk modulus at 20 °C


@dataclass(frozen=True)
class Syringe:
    """A barrel of inner diameter ``bore`` on prismatic ``joint`` (module docstring); ``liquid`` m³ at the start, the
    rest of the start volume air at ``pressure``. ``dead``: the hub's dead space (BD Luer tips about 0.07 mL)."""

    name: str
    joint: str
    bore: float
    stroke: float                        # m, the stopper's full travel (the graduated barrel length)
    x0: float = 0.0                      # m, the stopper's travel from the head at joint q = 0
    dead: float = 7.0e-8                 # m³
    liquid: float = 0.0                  # m³
    pressure: float = P_ATM              # Pa
    polytropic: float = 1.0
    bulk_modulus: float = WATER_K
    coulomb: float = 0.0                 # N, stopper friction
    viscous: float = 0.0                 # N·s/m
    v_stick: float = 1.0e-3              # m/s

    @property
    def area(self) -> float:
        return math.pi * self.bore ** 2 / 4.0

    def volume(self, x: float) -> float:
        return self.dead + self.area * max(x, 0.0)

    def node(self, x: float) -> Node:
        return Node(self.name, medium="mixed", volume=self.volume(x), liquid=self.liquid, radius=self.bore / 2.0,
                    pressure=self.pressure, polytropic=self.polytropic, bulk_modulus=self.bulk_modulus)


@dataclass(frozen=True)
class Vial:
    """A rigid closed chamber: ``volume`` m³, ``liquid`` of it at the start, the headspace at ``pressure``. Its head
    (the stopper end) points along ``axis`` in ``body``'s frame, or, with no body, ``up`` stays as given."""

    name: str
    volume: float
    liquid: float
    radius: float
    body: str | None = None
    axis: tuple[float, float, float] = (0.0, 0.0, 1.0)
    up: float = 1.0
    pressure: float = P_ATM
    bulk_modulus: float = WATER_K

    def node(self) -> Node:
        return Node(self.name, medium="mixed", volume=self.volume, liquid=self.liquid, radius=self.radius,
                    pressure=self.pressure, bulk_modulus=self.bulk_modulus, up=self.up)


@wp.struct
class SyringeSpec:
    node: wp.int32
    area: wp.float64
    x0: wp.float64
    dead: wp.float64
    coulomb: wp.float32
    viscous: wp.float32
    v_stick: wp.float32


def needle(diameter: float, length: float, viscosity: float = 1.0e-3, gas_viscosity: float = MU_AIR) -> dict:
    """A needle's (or a line's) port conductances, Poiseuille π·d⁴/(128·μ·L) for the liquid and for air (volume flow
    at the upstream pressure: the mean-pressure factor of compressible Poiseuille flow is taken as 1, ΔP << P).
    Example: ``Edge("needle", "port", "syringe", "vial", **needle(0.26e-3, 0.016), depth_b=0.005)``."""
    if min(diameter, length, viscosity, gas_viscosity) <= 0:
        raise ValueError("diameter, length and viscosities must be > 0")
    k = math.pi * diameter ** 4 / (128.0 * length)
    return {"conductance": k / viscosity, "gas_conductance": k / gas_viscosity}


def inertance(bore: float, diameter: float, length: float, density: float = 1000.0) -> float:
    """The plunger's added mass [kg] from the liquid column in a needle: ρ·L·A²/a (A the barrel's bore area, a the
    needle's). It is the column's kinetic energy ½·ρ·a·L·(A·v/a)² over ½·v²."""
    big, small = math.pi * bore ** 2 / 4.0, math.pi * diameter ** 2 / 4.0
    return density * length * big * big / small




@wp.func
def _response(w: wp.int32, i: wp.int32, v_new: wp.float64, dt: wp.float64, nodes: MixedNodes, ports: MixedPorts,
              pressure: wp.array2d[wp.float64]) -> wp.float64:
    """The pressure the plunger force uses: node i's implicit end-of-step pressure with its ports' far ends held,
    (κ·P' + dt·Σ s·g·P_far)/(κ + dt·Σ s·g) (P' closed at the new volume, below the vapour floor by its cavity/κ as in
    the core's solve; Σ over its ports' last conductances; P_far a reservoir's or the other node's pressure), at
    least P_VAPOUR. Holding the far ends at node i's own last pressure instead counted its steady outflow twice (the
    force 87 % high on a plunger pushed through 25 G, and a lag that rang the finger rig)."""
    liq = nodes.liquid[w, i]
    a = nodes.gas[w, i]
    c = nodes.gas_c[w, i]
    n = nodes.poly[i]
    kb = nodes.bulk[i]
    pr = nodes.p_ref[i]
    p1 = closure(v_new, liq, a, c, n, kb, pr)
    k = capacitance(liq, a, c, n, kb, p1, p1)
    num = k * p1 - cavity(v_new, liq, a, c, n, kb, pr, p1)
    den = k
    for e in range(ports.ea.shape[0]):
        far = ports.p_res[e]
        s = wp.float64(0.0)
        if ports.ea[e] == i:
            s = ports.sa[w, e]
            if ports.eb[e] >= 0:
                far = pressure[w, ports.eb[e]]
        if ports.eb[e] == i:
            s = ports.sb[w, e]
            far = pressure[w, ports.ea[e]]
        num = num + dt * s * ports.g[w, e] * far
        den = den + dt * s * ports.g[w, e]
    return wp.max(num / den, wp.float64(P_VAPOUR))


@wp.kernel
def syringe_apply(body_q: wp.array[wp.transform], body_qd: wp.array[wp.spatial_vector],
                  body_com: wp.array[wp.vec3], body_inv_mass: wp.array[wp.float32], dt: wp.float32,
                  joint_parent: wp.array[wp.int32], joint_child: wp.array[wp.int32], joint_X_p: wp.array[wp.transform],
                  joint_X_c: wp.array[wp.transform], joint_axis: wp.array[wp.vec3], joint_qd_start: wp.array[wp.int32],
                  n_syr: wp.int32, joint: wp.array2d[wp.int32], spec: wp.array[SyringeSpec], push: wp.array2d[float],
                  n_vial: wp.int32, vial_body: wp.array2d[wp.int32], vial_node: wp.array[wp.int32],
                  vial_axis: wp.array[wp.vec3], up_world: wp.vec3, atm: wp.float64, nodes: MixedNodes,
                  ports: MixedPorts, pressure: wp.array2d[wp.float64], volume: wp.array2d[wp.float64],
                  travel: wp.array2d[float], speed: wp.array2d[float], force: wp.array2d[float],
                  body_f: wp.array[wp.spatial_vector]):
    """One thread per world: each plunger's travel -> barrel volume and head direction, pressure -> force; each
    vial's head direction from its body."""
    w = wp.tid()
    for k in range(n_syr):
        s = spec[k]
        j = joint[w, k]
        par = joint_parent[j]
        ch = joint_child[j]
        xp = joint_X_p[j]
        if par >= 0:
            xp = wp.transform_multiply(body_q[par], xp)
        xc = wp.transform_multiply(body_q[ch], joint_X_c[j])
        ax = wp.normalize(wp.transform_vector(xp, joint_axis[joint_qd_start[j]]))
        anchor = wp.transform_get_translation(xc)
        com_c = com_world(body_q, body_com, ch)
        com_p = wp.vec3(0.0)
        x = float(s.x0) + wp.dot(anchor - wp.transform_get_translation(xp), ax)
        v = point_velocity(body_qd, com_c, ch, anchor)
        inv = body_inv_mass[ch]
        if par >= 0:
            com_p = com_world(body_q, body_com, par)
            v = v - point_velocity(body_qd, com_p, par, anchor)
            inv = inv + body_inv_mass[par]
        xd = wp.dot(v, ax)
        v_new = s.dead + s.area * wp.max(wp.float64(x), wp.float64(0.0))
        volume[w, s.node] = v_new
        nodes.up[w, s.node] = wp.float64(-wp.dot(ax, up_world))       # the head is at -axis
        p = _response(w, s.node, v_new, wp.float64(dt), nodes, ports, pressure)
        f = float((p - atm) * s.area) - push[w, k]
        fr = s.coulomb * wp.tanh(xd / s.v_stick) + s.viscous * xd
        if inv > 0.0:                                    # explicit friction: stop the slide at most, never reverse it
            cap = wp.abs(xd) / (inv * dt)
            fr = wp.clamp(fr, -cap, cap)
        f = f - fr
        add_wrench(body_f, ch, wrench_at(f * ax, anchor, com_c))
        if par >= 0:
            add_wrench(body_f, par, wrench_at(-f * ax, anchor, com_p))
        travel[w, k] = x
        speed[w, k] = xd
        force[w, k] = f
    for k in range(n_vial):
        b = vial_body[w, k]
        if b >= 0:
            head = wp.quat_rotate(wp.transform_get_rotation(body_q[b]), vial_axis[k])
            nodes.up[w, vial_node[k]] = wp.float64(wp.dot(wp.normalize(head), up_world))


def _spec(s: Syringe, node: int) -> SyringeSpec:
    out = SyringeSpec()
    out.node, out.area, out.x0, out.dead = node, s.area, s.x0, s.dead
    out.coulomb, out.viscous, out.v_stick = s.coulomb, s.viscous, s.v_stick
    return out


DEFAULT_SYRINGES = (Syringe("syringe", joint="syringe_plunger", bore=8.59e-3, stroke=0.06, x0=0.0523, liquid=2.8e-6),)
DEFAULT_PORTS = (Edge("nozzle", "port", "syringe", reservoir=P_ATM, **needle(0.26e-3, 0.016)),)


class Syringes(ForcePlugin):
    """Syringes, vials and needles (module docstring). The default config: a 3 mL barrel (BD, 8.59 mm bore) holding
    2.8 mL of water and 0.3 mL of air, a 25 G x 16 mm needle open to the air, on joint ``syringe_plunger``.

    Example (draw from an inverted vial through an 18 G fill needle)::

        Syringes(syringes=(Syringe("syr", "syr_plunger", bore=8.59e-3, stroke=0.06),),
                 vials=(Vial("vial", volume=10e-6, liquid=5e-6, radius=10.75e-3, body="vial"),),
                 ports=(Edge("ndl", "port", "syr", "vial", depth_b=0.005, **needle(0.84e-3, 0.038)),))
    """

    name = "syringe"
    summary = "syringes, vials and needles: liquid with an air pocket, plunger forces, air ledgers (GA syringe tasks)"

    @dataclass
    class Config:
        syringes: tuple[Syringe, ...] = DEFAULT_SYRINGES
        vials: tuple[Vial, ...] = ()
        ports: tuple[Edge, ...] = DEFAULT_PORTS
        openings: tuple[tuple[str, float], ...] = ()     # start openings (others 1)
        atmosphere: float = P_ATM

        def __post_init__(self) -> None:
            self.syringes = tuple(s if isinstance(s, Syringe) else Syringe(**s) for s in self.syringes)
            self.vials = tuple(v if isinstance(v, Vial) else Vial(**v) for v in self.vials)
            self.ports = tuple(e if isinstance(e, Edge) else Edge(**e) for e in self.ports)
            pairs = self.openings.items() if isinstance(self.openings, dict) else self.openings
            self.openings = tuple((str(k), float(v)) for k, v in pairs)
            net = self.network()                       # raises on a bad network
            edges = {e.name for e in net.edges}
            for name, value in self.openings:
                if name not in edges or not 0.0 <= value <= 1.0:
                    raise ValueError(f"openings: {name!r} must be a port and its value in [0, 1], got {value}")

        def network(self) -> Network:
            return Network([s.node(s.x0) for s in self.syringes] + [v.node() for v in self.vials], self.ports)

    def setup(self, ctx: Any) -> None:
        c, model, W = self.config, ctx.model, ctx.world_count
        net = c.network()
        self.fluid = net.compile(ctx.device, W)
        mixed = self.fluid.mixed
        S, V = len(c.syringes), len(c.vials)
        self.n_syr, self.n_vial = S, V
        jt = model.joint_type.numpy()
        joints = np.stack([ctx.joints(s.joint).numpy() for s in c.syringes], axis=1) if S else np.zeros((W, 1))
        for k, s in enumerate(c.syringes):
            if np.any(jt[joints[:, k].astype(int)] != int(newton.JointType.PRISMATIC)):
                raise PluginError(f"{self.label}: syringe {s.name!r} needs a prismatic joint, {s.joint!r} is not")
        self.joint = ctx.array(joints.astype(np.int32), wp.int32)
        self.spec = wp.array([_spec(s, net.node_index(s.name)) for s in c.syringes] or [SyringeSpec()],
                             dtype=SyringeSpec, device=ctx.device)
        bodies = np.stack([ctx.bodies(v.body).numpy() if v.body else np.full(W, -1) for v in c.vials], axis=1) \
            if V else np.full((W, 1), -1)
        self.vial_body = ctx.array(bodies.astype(np.int32), wp.int32)
        self.vial_node = ctx.array(np.array([net.node_index(v.name) for v in c.vials] or [0], np.int32), wp.int32)
        self.vial_axis = ctx.array(np.array([v.axis for v in c.vials] or [(0.0, 0.0, 1.0)], np.float32), wp.vec3)
        g = ctx.gravity
        self.up_world = wp.vec3(*(-g / np.linalg.norm(g) if np.linalg.norm(g) > 0 else (0.0, 0.0, 1.0)))
        start = self._start_volumes(model, joints)
        opening = self.fluid.opening.numpy()
        for name, value in c.openings:
            opening[:, net.edge_index(name)] = value
        self.fluid.opening.assign(opening)
        for k, a in self.fluid.state_arrays().items():
            self.add_state(k, a)
        for k, a in (("conductance", self.fluid._g), ("scale_a", mixed.ports.sa), ("scale_b", mixed.ports.sb)):
            self.add_state(k, a)                       # the last step's port g: the plunger force reads it
        shape = (W, max(S, 1))
        self.travel = self.add_state("travel", ctx.array(start, wp.float32))   # valid before the first step
        self.speed = self.add_state("speed", ctx.zeros(shape))
        self.force = self.add_state("force", ctx.zeros(shape))
        self.push = self.add_input("push", ctx.zeros(shape))
        self.add_input("opening", self.fluid.opening)

    def _start_volumes(self, model: newton.Model, joints: np.ndarray) -> np.ndarray:
        """Each world's barrels at their start travel (x0 + the joint's coordinate, returned [W, S]); the pocket holds
        the rest."""
        q, q_start = model.joint_q.numpy(), model.joint_q_start.numpy()
        vol, gas = self.fluid.volume.numpy(), self.fluid.mixed.gas.numpy()
        start = np.zeros((vol.shape[0], max(len(self.config.syringes), 1)), np.float32)
        for k, s in enumerate(self.config.syringes):
            i = self.fluid.net.node_index(s.name)
            x = s.x0 + q[q_start[joints[:, k].astype(int)]].astype(np.float64)
            start[:, k] = x
            vol[:, i] = s.dead + s.area * np.maximum(x, 0.0)
            v_l = s.liquid * (1.0 - (s.pressure - P_ATM) / s.bulk_modulus)
            if np.any(vol[:, i] < v_l * (1 - 1e-9)):
                raise PluginError(f"{self.label}: syringe {s.name!r} holds {s.liquid} m³ of liquid but its start "
                                  f"travel leaves {vol[:, i].min()} m³: raise x0 or lower liquid")
            gas[:, i] = s.pressure * np.maximum(vol[:, i] - v_l, 0.0)
        self.fluid.volume.assign(vol)
        self.fluid.volume_old.assign(vol)
        self.fluid.mixed.gas.assign(gas)
        return start

    def set_opening(self, port: str, values: Any) -> None:
        """Opening of ``port`` (a scalar or one per world, in [0, 1]); host copy, call between periods."""
        a = self.fluid.opening.numpy()
        a[:, self.fluid.net.edge_index(port)] = np.clip(np.asarray(values, np.float32), 0.0, 1.0)
        self.fluid.opening.assign(a)

    def air_ml(self, node: str) -> np.ndarray:
        """[W] mL of gas in a barrel or vial, at P_atm (GA's "air in the syringe")."""
        return self.fluid.mixed.gas.numpy()[:, self.fluid.net.node_index(node)] / P_ATM * 1e6

    def air_out_ml(self, port: str) -> np.ndarray:
        """[W] mL of gas (at P_atm) a port has carried out of its node a (GA's "air into the vein")."""
        return self.fluid.mixed.air_out_ml(port)

    def plunger_limits(self, model: newton.Model) -> dict[str, tuple[float, float]]:
        """Per syringe (world 0): (the needle damper's dt bound 2·I/c, the capped all-liquid spring's 2·sqrt(I/k))
        [s], I = the plunger's mass + its joint's armature, c = A²/G over its narrowest liquid port, k = A²·K/V at
        its start liquid (a full barrel when it starts empty)."""
        out = {}
        arm, mass = model.joint_armature.numpy(), model.body_mass.numpy()
        qd_start, child = model.joint_qd_start.numpy(), model.joint_child.numpy()
        for k, s in enumerate(self.config.syringes):
            j = int(self.joint.numpy()[0, k])
            inertia = float(arm[qd_start[j]] + mass[child[j]])
            g = [e.conductance for e in self.config.ports if s.name in (e.a, e.b) and e.conductance > 0]
            c = s.area ** 2 / min(g) if g else 0.0
            v_ref = s.dead + (s.liquid if s.liquid > 0 else s.area * s.stroke)
            spring = s.area ** 2 * s.bulk_modulus / v_ref
            out[s.name] = (2.0 * inertia / c if c > 0 else math.inf, 2.0 * math.sqrt(inertia / spring))
        return out

    def check_dt(self, dt: float) -> None:
        for name, (damper, spring) in self.plunger_limits(self.ctx.model).items():
            if dt > min(damper, spring):
                s = next(s for s in self.config.syringes if s.name == name)
                g = [e.conductance for e in self.config.ports if name in (e.a, e.b) and e.conductance > 0]
                need = max((s.area ** 2 / min(g)) * dt / 2.0 if g else 0.0,
                           s.area ** 2 * s.bulk_modulus / (s.dead + (s.liquid or s.area * s.stroke)) * dt * dt / 4.0)
                raise PluginError(f"{self.label}: syringe {name!r}'s plunger is unstable at dt = {dt:.3g} s (needle "
                                  f"damper bound {damper:.3g} s, capped-liquid spring bound {spring:.3g} s): give "
                                  f"joint {s.joint!r} an armature of at least {need:.3g} kg (the needle column's "
                                  "added mass, syringe.inertance(bore, needle_d, needle_L)) or lower the step")

    def apply(self, state: newton.State, forces: Forces, dt: float) -> None:
        m, f = self.ctx.model, self.fluid
        wp.launch(syringe_apply, dim=self.ctx.world_count,
                  inputs=[state.body_q, state.body_qd, m.body_com, m.body_inv_mass, float(dt), m.joint_parent,
                          m.joint_child, m.joint_X_p, m.joint_X_c, m.joint_axis, m.joint_qd_start, self.n_syr,
                          self.joint, self.spec, self.push, self.n_vial, self.vial_body, self.vial_node,
                          self.vial_axis, self.up_world, wp.float64(self.config.atmosphere), f.mixed.nodes,
                          f.mixed.ports, f.pressure, f.volume],
                  outputs=[self.travel, self.speed, self.force, forces.body_f], device=self.ctx.device)

    def update(self, state_in: newton.State, state_out: newton.State, dt: float) -> None:
        self.fluid.step(dt)

    def test_scene(self) -> Any:
        from alloy.plugins.library.syringe_laws import purge_scene  # noqa: PLC0415

        return purge_scene(self)

    def laws(self) -> list[Any]:
        from alloy.plugins.library.syringe_laws import syringe_laws  # noqa: PLC0415

        return syringe_laws(self)


__all__ = ["DEFAULT_PORTS", "DEFAULT_SYRINGES", "Syringe", "SyringeSpec", "Syringes", "Vial", "inertance", "needle",
           "syringe_apply"]
