# SPDX-License-Identifier: Apache-2.0
"""Which bodies form a rod chain for the direct solve (SCR-WP2-10).

A chain is an ordered run of bodies, each consecutive pair joined by one joint:
- Newton ROD joints from `ModelBuilder.add_rod`;
- a FIXED joint that welds an end body (a plug, a ferrule) to the rod's first or last segment.

The welded body must be in the chain. Solved separately (neighbours frozen), a stiff weld to a body of similar mass
converges as slowly as the rod's bending did: +541 % tip drop at one iteration against -1.5 % inside the chain
([internal note]).

Flat slot layout:
- rod r owns slots start[r] .. start[r] + count[r] - 1;
- body[slot] is that slot's body;
- joint[slot] joins body[slot - 1] and body[slot], with -1 at a rod's first slot.
A joint may point either way (its parent can be the later body).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ROD = 7      # newton.JointType.ROD
FIXED = 3    # newton.JointType.FIXED


@dataclass(frozen=True)
class RodTopology:
    body: np.ndarray
    joint: np.ndarray
    start: np.ndarray
    count: np.ndarray

    @property
    def rods(self) -> int:
        return int(len(self.start))

    @staticmethod
    def from_rods(rods: list[tuple[list[int], list[int]]]) -> RodTopology:
        """Chains given as (bodies, joints) with joints[k] joining bodies[k] and bodies[k + 1]."""
        body, joint, start, count = [], [], [], []
        for bodies, joints in rods:
            if len(bodies) < 2 or len(joints) != len(bodies) - 1:
                raise ValueError(f"an open chain needs >= 2 bodies and one joint fewer, got {len(bodies)} bodies, "
                                 f"{len(joints)} joints")
            start.append(len(body))
            count.append(len(bodies))
            body.extend(int(b) for b in bodies)
            joint.extend([-1, *(int(j) for j in joints)])
        return RodTopology(*(np.asarray(x, np.int32) for x in (body, joint, start, count)))

    @staticmethod
    def from_model(model) -> RodTopology:
        """Every maximal chain of enabled ROD joints in `model`. Each is extended at either end by a body welded there
        (FIXED) that has no other weld and no ROD joint. A rod graph with a junction, or a closed rod loop, raises:
        each needs a different solve."""
        jt, par, chi = (np.asarray(a.numpy()) for a in (model.joint_type, model.joint_parent, model.joint_child))
        en = np.asarray(model.joint_enabled.numpy())
        rod_nbr = _neighbours(jt, par, chi, en, ROD)
        bad = sorted(b for b, v in rod_nbr.items() if len(v) > 2)
        if bad:
            raise ValueError(f"rod junction at bodies {bad[:5]}: a graph rod needs a tree solve, not one chain")
        welds = _neighbours(jt, par, chi, en, FIXED)
        seen: set[int] = set()
        chains = []
        for b0 in sorted(rod_nbr):
            if b0 in seen or len(rod_nbr[b0]) != 1:
                continue
            bodies, joints = _walk(rod_nbr, b0)
            seen.update(bodies)
            _extend_welded_ends(bodies, joints, rod_nbr, welds)
            chains.append((bodies, joints))
        loops = sorted(set(rod_nbr) - seen)
        if loops:
            raise ValueError(f"closed rod loop at bodies {loops[:5]}: not one open chain")
        return RodTopology.from_rods(chains)


def _neighbours(jt, par, chi, en, kind: int) -> dict[int, list[tuple[int, int]]]:
    out: dict[int, list[tuple[int, int]]] = {}
    for j in np.flatnonzero((jt == kind) & en & (par >= 0)):
        p, c = int(par[j]), int(chi[j])
        out.setdefault(p, []).append((c, int(j)))
        out.setdefault(c, []).append((p, int(j)))
    return out


def _walk(nbr: dict[int, list[tuple[int, int]]], b0: int) -> tuple[list[int], list[int]]:
    bodies, joints, prev, cur = [b0], [], -1, b0
    while True:
        nxt = [(o, j) for o, j in nbr[cur] if o != prev]
        if not nxt:
            return bodies, joints
        o, j = nxt[0]
        bodies.append(o)
        joints.append(j)
        prev, cur = cur, o


def _extend_welded_ends(bodies: list[int], joints: list[int], rod_nbr, welds) -> None:
    for end in (0, -1):
        ends = [(o, j) for o, j in welds.get(bodies[end], []) if o not in rod_nbr and len(welds.get(o, [])) == 1]
        if len(ends) != 1:
            continue
        o, j = ends[0]
        if end == 0:
            bodies.insert(0, o)
            joints.insert(0, j)
        else:
            bodies.append(o)
            joints.append(j)


__all__ = ["FIXED", "ROD", "RodTopology"]
