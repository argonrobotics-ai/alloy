# SPDX-License-Identifier: Apache-2.0
"""The surface-rinsing ports (unofficial ports of Gym-Anything surface-rinsing-tasks-v1) as environments: seven tasks,
each on its port's own run, every world batched in one simulation.

| env | Gym-Anything task | what the policy moves |
|---|---|---|
| ``RinseSponge-v0`` | rinse-sponge | the hand on the sponge: target pose, velocity, grip |
| ``RinseDrainWash-v0`` | drain-wash-compartment | the hand on the drain levers: each lever's target and grip |
| ``RinseTomatoes-v0`` | rinse-tomatoes | the hand on the colander; the faucet's regulator |
| ``RinseFlushFloor-v0`` | flush-floor | the hose valve (kinematic): pose and velocity; the valve's opening |
| ``RinseGlassware-v0`` | rinse-glassware | the hand on six glasses: each one's target pose, velocity, grip |
| ``RinsePlates-v0`` | pre-rinse-plates | the hand on the plates and the spray valve; the valve's opening and balancer |
| ``RinseGarbageCan-v0`` | rinse-garbage-can | the hose valve (kinematic) and its opening; the hand on the can |

The hand is the ports' privileged scripted hand (``alloy.tasks.rinse_hand.PoseHand``: Cartesian impedance on held
bodies, no robot arm), so its proprio is the held bodies' pose and velocity. Success is the port's verdict per world:
every gate of the port after the settle (flush-floor: ``success_ga``, its gates named after the task's; the hands-free
gate reads the script's grip flag: ``docs/guide/envs.md`` lists what each port judges). The reward at the end is
the port's score (1 for a pass, a share of the progress otherwise). Water and soil films are not drawn by the camera
yet.

``make("RinseGlassware-v0", glasses=1, coarse=True)`` runs one glass on the port's coarse CPU test scene: a short
episode with the same checks, on a smaller scene than the task's six glasses (``glassware_task``).
"""

from __future__ import annotations

from typing import Any

from alloy.envs.ports.env import PortSpec
from alloy.envs.ports.movers import HandTargets, KinematicBody, Mover, PluginInput


class JointState:
    """Proprio only (no action): the q and qd of joints ``joints`` (label suffixes)."""

    def __init__(self, joints: tuple[str, ...]) -> None:
        self.joints, self.size = joints, 0

    def bind(self, sim: Any) -> None:
        import numpy as np  # noqa: PLC0415

        from alloy.envs.ports.movers import _match  # noqa: PLC0415

        m = sim.model
        labels, world = [str(lbl) for lbl in m.joint_label], m.joint_world.numpy()
        idx = np.stack([_match(labels, world, j, "joint") for j in self.joints], axis=1)
        self.q, self.dof = m.joint_q_start.numpy()[idx], m.joint_qd_start.numpy()[idx]
        self.low = self.high = np.zeros(0, np.float32)
        self.proprio_size = 2 * len(self.joints)

    def read(self, sim: Any) -> Any:
        import numpy as np  # noqa: PLC0415

        return np.zeros((self.q.shape[0], 0), np.float32)

    def write(self, sim: Any, a: Any) -> None:
        pass

    def proprio(self, sim: Any) -> Any:
        import numpy as np  # noqa: PLC0415

        q, qd = sim.state.joint_q.numpy()[self.q], sim.state.joint_qd.numpy()[self.dof]
        return np.concatenate([q, qd], axis=1).astype(np.float32)


# --- runs ----------------------------------------------------------------------------------------------------------


def run_sponge(worlds: int, device: str, seed: int) -> dict:
    from alloy.tasks import sponge_rinse_run  # noqa: PLC0415

    return sponge_rinse_run.run(worlds=worlds, seed=seed, device=device)


def run_drain(worlds: int, device: str, seed: int) -> Any:  # noqa: ARG001
    from alloy.tasks import rinse_drain  # noqa: PLC0415

    return rinse_drain.run(worlds=worlds, device=device)


def run_tomatoes(worlds: int, device: str, seed: int) -> Any:  # noqa: ARG001
    from alloy.tasks import rinse_tomatoes  # noqa: PLC0415

    return rinse_tomatoes.run(worlds=worlds, device=device)


def run_flush_floor(worlds: int, device: str, seed: int) -> dict:
    from alloy.tasks import rinse_flush_floor  # noqa: PLC0415

    return rinse_flush_floor.run(worlds=worlds, seed=seed, device=device)


COARSE_CHART = {"ds": 4.0e-3, "n_theta": 16}       # the glass charts of the port's CPU test (default 2 mm, 32)
COARSE_PACE = {"rinse_s": 2.0, "pace": 0.6}        # and its script: 2 s rinse dunks, moves at 0.6 of their time


def glassware_task(glasses: int = 6, coarse: bool = False) -> Any:
    """The port's settings: the task's six glasses, or a smaller scene that is not the task. ``glasses`` (1 to 6)
    rinses only that many; ``coarse=True`` is the port's CPU test scene and script (4 mm chart cells, 16 around a
    glass, 2 s rinse dunks, faster moves). Example: ``glassware_task(1, True).station.count`` is 1."""
    from alloy.envs.errors import EnvError  # noqa: PLC0415
    from alloy.tasks.rinse_glass_scene import GlassStation  # noqa: PLC0415
    from alloy.tasks.rinse_glassware import Glassware  # noqa: PLC0415
    from alloy.tasks.rinse_sink import BOWLS  # noqa: PLC0415

    if isinstance(glasses, bool) or not isinstance(glasses, int) or not 1 <= glasses <= 6:
        raise EnvError(f"RinseGlassware-v0: glasses={glasses!r} is not a whole number from 1 to 6",
                       fix="pass glasses=1 ... 6 (6 is the task; fewer is a smaller scene)", field="glasses")
    station = GlassStation(count=glasses, weir=BOWLS, **(COARSE_CHART if coarse else {}))
    return Glassware(station=station, **(COARSE_PACE if coarse else {}))


def run_glassware(worlds: int, device: str, seed: int, glasses: int = 6, coarse: bool = False) -> Any:  # noqa: ARG001
    from alloy.tasks import rinse_glassware  # noqa: PLC0415

    return rinse_glassware.run(glassware_task(glasses, coarse), worlds=worlds, device=device)


def run_plates(worlds: int, device: str, seed: int) -> Any:  # noqa: ARG001
    from alloy.tasks import rinse_plates  # noqa: PLC0415

    return rinse_plates.run(worlds=worlds, device=device)


def run_garbage_can(worlds: int, device: str, seed: int) -> dict:
    from alloy.tasks import can_wash_tip  # noqa: PLC0415

    return can_wash_tip.run_empty(worlds=worlds, seed=seed, device=device)


# --- verdicts ------------------------------------------------------------------------------------------------------


def _row_success(key: str = "success"):  # noqa: ANN202
    return lambda r, w: bool(r["worlds"][w][key])


def _row_score(r: Any, w: int) -> float:
    row = r["worlds"][w]
    return 1.0 if row.get("success") else float(row.get("score") or 0.0)


def _res_success(r: Any, w: int) -> bool:
    return bool(r.success[w])


def _res_score(r: Any, w: int) -> float:
    return 1.0 if r.success[w] else float(r.score[w] or 0.0)


def _drain_mover() -> Mover:
    from alloy.tasks.rinse_sink import BOWLS, lever_joint  # noqa: PLC0415

    return Mover(PluginInput("target", -3.2, 3.2, "lever_hand"), PluginInput("grip", 0.0, 1.0, "lever_hand"),
                 JointState(tuple(lever_joint(b) for b in BOWLS)))



SPECS = [
    PortSpec("RinseSponge-v0", "surface_rinsing/rinse-sponge", run_sponge, lambda: Mover(HandTargets()),
             _row_success(), score=_row_score, batched=True,
             summary="rinse the soap out of a sponge under the tap and wring it"),
    PortSpec("RinseDrainWash-v0", "surface_rinsing/drain-wash-compartment", run_drain, _drain_mover,
             _res_success, score=_res_score, batched=True,
             summary="drain a three-compartment sink's wash bowl with its lever"),
    PortSpec("RinseTomatoes-v0", "surface_rinsing/rinse-tomatoes", run_tomatoes,
             lambda: Mover(HandTargets(), PluginInput("regulator")), _res_success, score=_res_score, batched=True,
             summary="rinse dusty tomatoes in a colander under the faucet"),
    PortSpec("RinseFlushFloor-v0", "surface_rinsing/flush-floor", run_flush_floor,
             lambda: Mover(KinematicBody("hose_valve"), PluginInput("valve")), _row_success("success_ga"),
             score=lambda r, w: 1.0 if r["worlds"][w]["success_ga"] else float(r["worlds"][w].get("score") or 0.0),
             batched=True, summary="flush a wash bay's floor to its drain with a hose"),
    PortSpec("RinseGlassware-v0", "surface_rinsing/rinse-glassware", run_glassware, lambda: Mover(HandTargets()),
             _res_success, score=_res_score, batched=True,
             summary="rinse and sanitize six glasses and set them inverted to drain"),
    PortSpec("RinsePlates-v0", "surface_rinsing/pre-rinse-plates", run_plates,
             lambda: Mover(HandTargets(), PluginInput("grip", 0.0, 1.0, "balancer"), PluginInput("valve")),
             _res_success, score=_res_score, batched=True,
             summary="pre-rinse soiled plates with a spray valve and rack them"),
    PortSpec("RinseGarbageCan-v0", "surface_rinsing/rinse-garbage-can", run_garbage_can,
             lambda: Mover(KinematicBody("hose_valve"), PluginInput("valve"), HandTargets()), _row_success(),
             score=_row_score, batched=True,
             summary="hose out a garbage can, tip it to dump the water and set it upright"),
]

__all__ = ["SPECS", "run_drain", "run_flush_floor", "run_garbage_can", "run_glassware", "run_plates",
           "run_sponge", "run_tomatoes"]
