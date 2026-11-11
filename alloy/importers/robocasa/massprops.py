# SPDX-License-Identifier: Apache-2.0
"""Mass, centre of mass and inertia of a body from its visual parts, computed by MuJoCo's mesh inertia.

Each part gets a density from its material. A closed (watertight) part is a solid (``inertia="exact"``); an open
part is a thin shell of thickness ``shell_t`` (``inertia="shell"``, areal density = density x thickness). MuJoCo
compiles a throwaway model holding only those parts and reports the body's mass, centre of mass and principal
inertia, which the writer copies into ``<inertial>``.

Example::

    >>> import numpy as np
    >>> from alloy.importers.robocasa.massprops import PartMass, body_inertial, unit_cube_faces, unit_cube_vertices
    >>> box = PartMass("cube", unit_cube_vertices(0.1), unit_cube_faces(), density=1000.0, closed=True)
    >>> round(body_inertial([box]).mass, 6)                # 0.1 m cube of water
    1.0
"""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from alloy.cards import geometry as geo

log = logging.getLogger("alloy.importers.robocasa")


@dataclass(frozen=True)
class PartMass:
    """One visual part in the body frame: vertices (m), faces, density (kg/m^3), solid or shell."""

    name: str
    vertices: np.ndarray
    faces: np.ndarray
    density: float
    closed: bool
    shell_t: float = 0.002          # m, wall thickness used when the part is open


@dataclass(frozen=True)
class Inertial:
    """Mass properties in the body frame: ``quat`` (w, x, y, z) rotates the principal axes into the body frame."""

    mass: float
    pos: tuple[float, float, float]
    quat: tuple[float, float, float, float]
    diaginertia: tuple[float, float, float]
    part_mass: dict[str, float]


def unit_cube_vertices(edge: float) -> np.ndarray:
    """Corners of an axis-aligned cube of side ``edge`` centred at the origin (for examples and tests)."""
    c = np.array([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)], dtype=np.float64)
    return (c - 0.5) * edge


def unit_cube_faces() -> np.ndarray:
    """Outward triangles for ``unit_cube_vertices``."""
    return np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
                     [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]], dtype=np.int64)


def _part_xml(i: int, p: PartMass, folder: Path) -> tuple[str, str]:
    path = folder / f"part{i}.obj"
    geo.write_obj(path, geo.MeshData(np.asarray(p.vertices, np.float64), np.asarray(p.faces, np.int64)))
    if p.closed:
        asset = f'<mesh name="m{i}" file="{path.name}" inertia="exact"/>'
        geom = f'<geom type="mesh" mesh="m{i}" density="{p.density!r}"/>'
    else:
        asset = f'<mesh name="m{i}" file="{path.name}" inertia="shell"/>'
        geom = f'<geom type="mesh" mesh="m{i}" density="{p.density * p.shell_t!r}"/>'
    return asset, geom


def _compile(parts: list[PartMass], folder: Path):
    """A throwaway model with one body per part (part ``i`` is body ``i + 1``)."""
    import mujoco  # noqa: PLC0415 - heavy import, only when mass properties are asked for

    xml = [_part_xml(i, p, folder) for i, p in enumerate(parts)]
    bodies = "".join(f'<body name="p{i}">{g}</body>' for i, (_, g) in enumerate(xml))
    text = (f'<mujoco><compiler meshdir="{folder}" boundmass="0" boundinertia="0"/>'
            f'<asset>{"".join(a for a, _ in xml)}</asset><worldbody>{bodies}</worldbody></mujoco>')
    try:
        return mujoco.MjModel.from_xml_string(text)
    except ValueError as exc:
        raise ValueError(f"MuJoCo cannot compute mass properties of {[p.name for p in parts]}: {exc}. "
                         "Repair or drop the degenerate part") from exc


def _combine(mass: np.ndarray, com: np.ndarray, quat: np.ndarray, diag: np.ndarray) -> tuple:
    """Total mass, centre of mass and inertia tensor (about that centre, body frame) of several bodies."""
    from alloy.importers.robocasa.mjcf_in import quat_matrix  # noqa: PLC0415 - avoid a cycle at import

    total = float(mass.sum())
    c = (mass[:, None] * com).sum(axis=0) / total
    tensor = np.zeros((3, 3))
    for m_i, x_i, q_i, d_i in zip(mass, com, quat, diag):
        r = quat_matrix(q_i)
        d = x_i - c
        tensor += r @ np.diag(d_i) @ r.T + m_i * (float(d @ d) * np.eye(3) - np.outer(d, d))
    return total, c, tensor


def _principal(tensor: np.ndarray) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """(diag inertia, quat w-x-y-z of the principal axes) of a symmetric inertia tensor."""
    vals, vecs = np.linalg.eigh(0.5 * (tensor + tensor.T))
    if np.linalg.det(vecs) < 0:
        vecs[:, 2] *= -1.0
    import mujoco  # noqa: PLC0415

    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, vecs.reshape(-1))
    return tuple(float(v) for v in np.maximum(vals, 0.0)), tuple(float(x) for x in q)


def body_inertial(parts: list[PartMass]) -> Inertial:
    """Mass properties of the union of ``parts`` (overlapping parts count twice, as MuJoCo geoms do)."""
    if not parts:
        raise ValueError("no parts: a body needs at least one visual part to get a mass")
    with tempfile.TemporaryDirectory(prefix="robocasa_mass_") as tmp:
        m = _compile(parts, Path(tmp))
    idx = np.arange(1, len(parts) + 1)
    total, com, tensor = _combine(m.body_mass[idx], m.body_ipos[idx] + m.body_pos[idx], m.body_iquat[idx],
                                  m.body_inertia[idx])
    if not total > 0:
        raise ValueError(f"parts {[p.name for p in parts]} have zero mass; check densities and mesh volumes")
    diag, quat = _principal(tensor)
    masses = {p.name: float(m.body_mass[i + 1]) for i, p in enumerate(parts)}
    return Inertial(total, tuple(float(x) for x in com), quat, diag, masses)


__all__ = ["Inertial", "PartMass", "body_inertial", "unit_cube_faces", "unit_cube_vertices"]
