# SPDX-License-Identifier: Apache-2.0
"""Contacts of the truck and the loads with the structures GA watches (SCR-WP8-12 F5), for the episode's contact
episodes, the loads' floor tracks and the rack beams a pallet sits on.

Once per control period the MuJoCo Warp solver's contacts are exported through Newton (``update_contacts``, with
forces); each contact's two shapes are classed by label:
- the mover: ``"truck"`` for a shape on one of the truck's bodies, ``"load:<name>"`` for a watched load's;
- the structure: the class of the first ``watch`` prefix its label starts with (``{"rack_": "rack", ...}``).
Per sample and per (mover, structure) the largest contact force is kept with its two shape labels and its normal;
``episodes`` groups each series into GA's contact episodes (``contact_episodes_from_series``). The onset speeds are
read from the sample before the contact's first one: by the end of the period in which a body meets a structure it
has already been stopped (WP13). A sample is the state at the end of a control period, so a contact that starts and
ends between two samples is missed (GA samples at its own tick). One world only.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np

from alloy._vendor import newton
from alloy.tasks.forklift_ga_types import ContactEpisode, contact_episodes_from_series

TRUCK_BODIES = ("forklift", "axle", "mast_outer", "mast_int", "mast_inner", "carriage", "sideshift")


def _short(label: Any) -> str:
    return str(label).rsplit("/", 1)[-1]


def _solver(sim: Any) -> Any:
    parts = getattr(sim, "parts", None)
    if parts is not None and getattr(parts, "solvers", None):
        return next(iter(parts.solvers.values()))
    from alloy.integration.stepper import entry_solvers  # noqa: PLC0415

    return next(iter(entry_solvers(sim.stepper).values()))


class ContactWatch:
    """Classifies one world's contacts per control period (world 0 of the simulation)."""

    def __init__(self, sim: Any, watch: dict[str, str], loads: dict[str, str] | None = None,
                 rename: dict[str, str] | None = None) -> None:
        self.sim, self.watch = sim, dict(watch)
        m = sim.model
        if m.world_count > 1:
            raise ValueError(f"ContactWatch reads one world; the simulation has {m.world_count}")
        self.solver = _solver(sim)
        n = int(self.solver.mjw_data.naconmax)
        self.contacts = newton.Contacts(n, 0, requested_attributes={"force"}, device=m.device)
        body_label = [_short(x) for x in m.body_label]
        shape_body = m.shape_body.numpy()
        rename = rename or {}                               # rig shape label -> GA's geom label (e.g. a nose wall)
        self.shape_label = [rename.get(_short(x), _short(x)) for x in m.shape_label]
        loads = loads or {}
        by_body = {b: f"load:{name}" for name, b in loads.items()}
        self.mover = [None] * len(shape_body)
        self.structure = [None] * len(shape_body)
        for s, b in enumerate(shape_body):
            body = body_label[b] if b >= 0 else ""
            if body in TRUCK_BODIES:
                self.mover[s] = "truck"
            elif body in by_body:
                self.mover[s] = by_body[body]
            self.structure[s] = self._class(self.shape_label[s])
        self.body_of = shape_body
        self.structure_bodies = sorted({int(b) for s, b in enumerate(shape_body) if self.structure[s] and b >= 0})
        self.touch: dict[str, set[str]] = {}              # mover -> every structure shape it touches (last read)

    def _class(self, label: str) -> str | None:
        for prefix, cls in self.watch.items():
            if label.startswith(prefix):
                return cls
        return None

    def read(self) -> dict[str, dict[str, tuple[float, str, str, np.ndarray, int]]]:
        """{mover: {structure: (largest force N, mover shape, structure shape, normal toward the mover, structure
        body)}} for the current state."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.solver.update_contacts(self.contacts)
        count = int(self.contacts.rigid_contact_count.numpy()[0])
        self.touch = {}
        if count == 0:
            return {}
        s0 = self.contacts.rigid_contact_shape0.numpy()[:count]
        s1 = self.contacts.rigid_contact_shape1.numpy()[:count]
        normal = self.contacts.rigid_contact_normal.numpy()[:count]
        force = np.linalg.norm(self.contacts.force.numpy()[:count, :3], axis=1)
        out: dict[str, dict[str, tuple]] = {}
        for k in range(count):
            if force[k] <= 0.0:                            # Newton's gap reports shapes near, not touching
                continue
            for a, b, sign in ((s0[k], s1[k], -1.0), (s1[k], s0[k], 1.0)):
                if a < 0 or b < 0:
                    continue
                mover, cls = self.mover[a], self.structure[b]
                if mover is None or cls is None or self.mover[b] == mover:
                    continue
                self.touch.setdefault(mover, set()).add(self.shape_label[b])
                best = out.setdefault(mover, {}).get(cls)
                if best is None or force[k] > best[0]:
                    out[mover][cls] = (float(force[k]), self.shape_label[a], self.shape_label[b],
                                       sign * normal[k].astype(np.float64), int(self.body_of[b]))
        return out

    def episodes(self, t: np.ndarray, rows: list[dict[str, Any]], io: Any) -> tuple[ContactEpisode, ...]:
        """GA's contact episodes from the sampled contacts, the onset speeds from the movers' velocities."""
        pairs = sorted({(m, c) for r in rows for m, per in r["contacts"].items() for c in per})
        out: list[ContactEpisode] = []
        for mover, cls in pairs:
            hits = [r["contacts"].get(mover, {}).get(cls) for r in rows]
            force = np.array([h[0] if h else 0.0 for h in hits])
            before = [rows[max(i - 1, 0)] for i in range(len(rows))]          # the approach, before the impact
            speed = np.array([self._speed(mover, r) for r in before])
            normal = np.array([self._approach(mover, r, h) if h else 0.0 for r, h in zip(before, hits, strict=True)])
            peak = int(np.argmax(force))
            parts = (hits[peak][1], hits[peak][2]) if hits[peak] else ()
            out += contact_episodes_from_series(mover, cls, t, force, speed, normal, parts=parts)
        return tuple(out)

    def beams(self, row: dict[str, Any], mover: str = "load:carried") -> tuple[str, ...]:
        """Rack beam shape labels the carried pallet touches in ``row``."""
        return tuple(sorted(label for label in row["touch"].get(mover, ()) if "_beam" in label))

    @staticmethod
    def _speed(mover: str, row: dict[str, Any]) -> float:
        if mover == "truck":
            return abs(float(row["speed"]))
        return float(np.linalg.norm(row.get(f"vel:{mover[5:]}", np.zeros(3))))

    @staticmethod
    def _approach(mover: str, row: dict[str, Any], hit: tuple) -> float:
        """Approach speed along the contact normal (rig velocities; the normal points toward the mover)."""
        if mover == "truck":
            v = np.asarray(row.get("vel:truck", np.zeros(3)))
        else:
            v = np.asarray(row.get(f"vel:{mover[5:]}", np.zeros(3)))
        v_s = row.get("vel_body", {}).get(hit[4], np.zeros(3))
        return max(-float((v - v_s) @ hit[3]), 0.0)


__all__ = ["ContactWatch", "TRUCK_BODIES"]
